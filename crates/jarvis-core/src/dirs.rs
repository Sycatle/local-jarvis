use std::path::PathBuf;

use directories::ProjectDirs;

fn project() -> ProjectDirs {
    ProjectDirs::from("local", "", "jarvis")
        .expect("could not resolve XDG project directories for jarvis")
}

pub fn config_dir() -> PathBuf {
    project().config_dir().to_path_buf()
}

pub fn data_dir() -> PathBuf {
    project().data_dir().to_path_buf()
}

pub fn cache_dir() -> PathBuf {
    project().cache_dir().to_path_buf()
}

pub fn state_dir() -> PathBuf {
    project()
        .state_dir()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| project().data_dir().to_path_buf())
}

pub fn config_file() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}
