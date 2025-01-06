use kalosm_language_model::Embedder;
use rbert::*;
use tokio::main;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut bert = Bert::new().await?;
    let sentences = [
        "Cats are cool",
        "The geopolitical situation is dire",
        "Pets are great",
        "Napoleon was a tyrant",
        "Napoleon was a great general",
        "123 Smith Street, Liverpool NSW",
        "Unit 3 , 123 Smith St, Liverpool, NSW"
    ];
    let embeddings = bert.embed_batch(sentences).await?;
    println!("embeddings {:?}", embeddings);

    let first_embed = embeddings.first().unwrap();

    let tensor = first_embed.vector();

    println!("{:?}",tensor);


    // Find the cosine similarity between the first two sentences
    let mut similarities = vec![];
    let n_sentences = sentences.len();
    for (i, e_i) in embeddings.iter().enumerate() {
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j).unwrap();
            let cosine_similarity = e_j.cosine_similarity(e_i);
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities.iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())
}