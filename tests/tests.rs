// Compiletest documentation:
//
// https://rustc-dev-guide.rust-lang.org/tests/compiletest.html
// https://rustc-dev-guide.rust-lang.org/tests/directives.html#assembly
// https://rustc-dev-guide.rust-lang.org/tests/ui.html#controlling-passfail-expectations
#![cfg(not(target_os = "wasi"))]

#[allow(dead_code)]
#[cfg_attr(miri, ignore)]
fn run_mode(mode: &'static str, custom_dir: Option<&'static str>) {
    let mut config = compiletest_rs::Config::default();
    let cfg_mode = mode.parse().expect("Invalid mode");

    config.mode = cfg_mode;

    let dir = custom_dir.unwrap_or(mode);
    config.src_base = std::path::PathBuf::from(format!("tests/{}", dir));
    config.target_rustcflags = Some("-L target/debug -L target/debug/deps".to_string());
    config.clean_rmeta(); // If your tests import the parent crate, this helps with E0464
    config.clean_rlib();
    config.strict_headers = true;
    config.llvm_filecheck = Some(
        // Set your local path to `FileCheck` as this environment variable
        // unless it's already present in $PATH from installing LLVM with
        // `-DLLVM_INSTALL_UTILS=ON` in cmake.
        std::env::var("FILECHECK")
            .unwrap_or("FileCheck".to_string())
            .into(),
    );

    compiletest_rs::run_tests(&config);
}

#[test]
#[cfg_attr(miri, ignore)]
fn compile_test() {
    // run_mode("compile-fail", None);
    // run_mode("run-pass", None);
    #[cfg(feature = "_assembly_x86")]
    run_mode("assembly", Some("assembly/x86"));
}
