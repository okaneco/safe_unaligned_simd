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
        std::env::var("FILECHECK")
            /* Comment out this line if using the following solutions */
            .unwrap_or("llvmbuild/bin/FileCheck".to_string())
            /* Uncomment out this line if using locally built LLVM */
            // .unwrap_or("FileCheck".to_string())
            /* Uncomment out this line if using `pip install filecheck`*/
            // .unwrap_or("filecheck".to_string())
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
