//! Skill registry + built-in skills (system control + media).

// Allow the `#[skill]` proc-macro to emit `::jarvis_skills::Skill` paths from
// within this crate itself.
extern crate self as jarvis_skills;

pub mod capabilities;
pub mod media;
pub mod rag;
pub mod registry;
pub mod system;

pub use capabilities::Capabilities;
pub use jarvis_skills_macros::skill;
pub use media::MediaSkills;
pub use rag::IndexDocsSkill;
pub use registry::{Skill, SkillRegistry};
pub use system::SystemSkills;
