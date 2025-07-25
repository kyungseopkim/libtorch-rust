use libtorch_rust::simple_test::run_basic_tests;

fn main() {
    match run_basic_tests() {
        Ok(()) => {
            println!("\n🎊 All tests completed successfully!");
            std::process::exit(0);
        }
        Err(e) => {
            println!("\n❌ Tests failed with error: {:?}", e);
            std::process::exit(1);
        }
    }
}