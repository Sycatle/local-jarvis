//! Proc-macro `#[skill(name = "...", description = "...")]`.
//!
//! Applied to an `impl` block that contains:
//!
//! - `async fn run(&self, args: ArgsType) -> serde_json::Value` (required)
//! - `fn capabilities(&self) -> jarvis_skills::Capabilities` (optional)
//!
//! Generates a `jarvis_skills::Skill` trait implementation:
//!
//! - `name()` / `description()` from the attribute literals.
//! - `parameters()` via `schemars::schema_for!(ArgsType)`.
//! - `invoke()` deserialises the JSON args and dispatches to `run`.
//! - `capabilities()` calls into the inline method if provided; otherwise
//!   defaults to `Capabilities::none()`.
//!
//! The inherent `run` and `capabilities` methods are kept on the type so they
//! remain callable directly (e.g. for unit tests).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, FnArg, ImplItem, ItemImpl, LitStr, PatType, Token, Type,
};

struct SkillAttr {
    name: LitStr,
    description: LitStr,
}

impl Parse for SkillAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name: Option<LitStr> = None;
        let mut description: Option<LitStr> = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: LitStr = input.parse()?;
            match key.to_string().as_str() {
                "name" => name = Some(value),
                "description" => description = Some(value),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[skill] attribute key `{other}` (expected name | description)"
                        ),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(Self {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "missing `name = \"...\"`"))?,
            description: description
                .ok_or_else(|| syn::Error::new(input.span(), "missing `description = \"...\"`"))?,
        })
    }
}

#[proc_macro_attribute]
pub fn skill(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as SkillAttr);
    let impl_block = parse_macro_input!(item as ItemImpl);

    let self_ty = &impl_block.self_ty;
    let args_ty = match extract_args_type(&impl_block) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };
    let has_caps = impl_block.items.iter().any(|item| {
        matches!(item, ImplItem::Fn(m)
            if m.sig.ident == "capabilities" && m.sig.asyncness.is_none())
    });

    let name = attr.name;
    let description = attr.description;
    let caps_call = if has_caps {
        quote! { <Self>::capabilities(self) }
    } else {
        quote! { ::jarvis_skills::Capabilities::none() }
    };

    let generated: TokenStream2 = quote! {
        #impl_block

        #[::async_trait::async_trait]
        impl ::jarvis_skills::Skill for #self_ty {
            fn name(&self) -> &'static str { #name }
            fn description(&self) -> &'static str { #description }
            fn parameters(&self) -> ::serde_json::Value {
                let schema = ::schemars::schema_for!(#args_ty);
                ::serde_json::to_value(schema).unwrap_or_else(|_| ::serde_json::json!({"type": "object"}))
            }
            fn capabilities(&self) -> ::jarvis_skills::Capabilities {
                #caps_call
            }
            async fn invoke(&self, args: ::serde_json::Value) -> ::serde_json::Value {
                match ::serde_json::from_value::<#args_ty>(args) {
                    Ok(parsed) => Self::run(self, parsed).await,
                    Err(e) => ::serde_json::json!({
                        "ok": false,
                        "error": format!("invalid arguments: {}", e),
                    }),
                }
            }
        }
    };

    generated.into()
}

/// Locate `async fn run(&self, args: ArgsType) -> ...` in the impl block and
/// return its `ArgsType`. We accept any second parameter name, so callers can
/// write `args`, `_args`, `input`, etc.
fn extract_args_type(impl_block: &ItemImpl) -> syn::Result<Type> {
    for item in &impl_block.items {
        let ImplItem::Fn(method) = item else { continue };
        if method.sig.ident != "run" {
            continue;
        }
        if method.sig.asyncness.is_none() {
            return Err(syn::Error::new_spanned(
                &method.sig,
                "#[skill]: `run` must be `async`",
            ));
        }
        let mut inputs = method.sig.inputs.iter();
        let receiver = inputs.next();
        let Some(FnArg::Receiver(_)) = receiver else {
            return Err(syn::Error::new_spanned(
                &method.sig,
                "#[skill]: `run` must take `&self` as its first parameter",
            ));
        };
        let args_arg = inputs.next().ok_or_else(|| {
            syn::Error::new_spanned(
                &method.sig,
                "#[skill]: `run` must take an `args: ArgsType` parameter",
            )
        })?;
        let FnArg::Typed(PatType { ty, .. }) = args_arg else {
            return Err(syn::Error::new_spanned(
                args_arg,
                "#[skill]: second parameter of `run` must be `args: ArgsType`",
            ));
        };
        return Ok((**ty).clone());
    }
    Err(syn::Error::new_spanned(
        impl_block,
        "#[skill]: impl block must contain `async fn run(&self, args: ArgsType) -> serde_json::Value`",
    ))
}
