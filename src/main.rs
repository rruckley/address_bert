// use kalosm_language_model::Embedder;
use rbert::*;
// use tokio::main;

use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, ScalarQuantizationBuilder,
    SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};

const QDRANT_KEY : &str = "ZXcfju_BKbEFzRcBdEDkZxeDyEf9y9WjuPcVI8JAQi5qZSwMlhhlfA";
const QDRANT_HOST : &str = "https://7713d7e3-b4f5-4ac5-b8db-a07f60e2ece7.europe-west3-0.gcp.cloud.qdrant.io";
const QDRANT_COLLECTION : &str = "address";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut bert = Bert::new().await?;
    let sentences = [
        "60 Ocean Beach Road,SORRENTO,VIC,3943",
        "KARDINYA PARK, Shop 14, 15 SOUTH STREET,KARDINYA,WA,6163",
        "STOCKLAND GLENDALE, SHOP 14A, 387 LAKE ROAD,GLENDALE,NSW,2285",
        "Shop 5, 64 GLADESVILLE ROAD,HUNTERS HILL,NSW,2110",
        "SHOP 22, 75 LYNBROOK BOULEVARD,Lynbrook,VIC,3975",
        "HALLETT COVE SHOP, Shop 57, 246 LONSDALE ROAD,HALLETT COVE,SA,5158",
        "Shop GM87, Colonnades S/C, 0 Beach Road,NOARLUNGA CENTRE,SA,5168",
    ];
    let embeddings = bert.embed_batch(sentences).await?;
    println!("embeddings {:?}", embeddings);

    let first_embed = embeddings.first().unwrap();

    let tensor = first_embed.vector();

    // println!("{:?}",tensor);

    let client = Qdrant::from_url(QDRANT_HOST)
        .api_key(QDRANT_KEY)
        .
        .build()?;

    let health = client.health_check().await;
    match health {
        Ok(o) => println!("Health ok"),
        Err(e) => println!("Failed: {}",e),
    }

    // client
    // .create_collection(
    //     CreateCollectionBuilder::new(QDRANT_COLLECTION)
    //         .vectors_config(VectorParamsBuilder::new(384, Distance::Cosine))
    // ).await?;

    // for (i, e_i) in embeddings.iter().enumerate() {
    //     let e_j = embeddings.get(0).unwrap();
    //     let e_v = e_j.vector();
    //     println!("E_V: {}",e_v.to_string());
    // }
    // Find the cosine similarity between the first two sentences
    // let mut similarities = vec![];
    // let n_sentences = sentences.len();
    // for (i, e_i) in embeddings.iter().enumerate() {
    //     for j in (i + 1)..n_sentences {
    //         let e_j = embeddings.get(j).unwrap();
    //         let cosine_similarity = e_j.cosine_similarity(e_i);
    //         similarities.push((cosine_similarity, i, j))
    //     }
    // }
    // similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    // for &(score, i, j) in similarities.iter() {
    //     println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    // }

    Ok(())
}