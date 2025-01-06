
use candle_core::{DType,Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::distilbert::{Config,DistilBertModel};
use log::info;
use tokenizers::{Tokenizer,TokenizerImpl};
use hf_hub::{api::sync::Api, Repo, RepoType};
use anyhow::{Error, Result};

fn build_model() -> Result<(DistilBertModel, Tokenizer)> {

    const MODEL : &str = "";

    let device = Device::Cpu;

    let model_id = MODEL.to_string();

    let revision = "main".to_string();

    let repo = Repo::with_revision(model_id, RepoType::Model, revision);

    let api = Api::new()?;

    let api = api.repo(repo);

    let config_file = api.get("config.json")?;

    let tokenizer_file = api.get("tokenizer.json")?;

    let config = std::fs::read_to_string(config_file)?;

    let mut config: Config = serde_json::from_str(&config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(Error::msg)?;

    let vb = match api.get("model.safetensors") {
        Ok(path) => {
            unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }
        },
        Err(_) => {

        }
    }?;

    let model = DistilBertModel::load(vb, &config)?;

    Ok((model,tokenizer)) 
}

fn embed() -> Result<()> {
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let pkg = env!("CARGO_PKG_NAME");
    let ver = env!("CARGO_PKG_VERSION");

    info!("Starting {pkg} v{ver}");

     

    Ok(())
}
