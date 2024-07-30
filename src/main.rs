
use candle_core::{Device, Result, Tensor};
use log::info;

fn main() -> Result<()> {
    env_logger::init();

    let pkg = env!("CARGO_PKG_NAME");
    let ver = env!("CARGO_PKG_VERSION");

    info!("Starting {pkg} v{ver}");

    Ok(())
}
