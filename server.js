const express = require("express");
const tf = require("@tensorflow/tfjs");
const use = require("@tensorflow-models/universal-sentence-encoder");
const cors = require("cors");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

let model;

app.get("/",(req,res)=>{
  res.send("hello working and running")
});

// Load Universal Sentence Encoder
(async () => {
  console.log("Loading Universal Sentence Encoder...");
  model = await use.load();
  console.log("Model loaded!");
})();

app.post("/embed", async (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: "Missing text in request body." });
  }

  try {
    const embeddings = await model.embed([text]);
    const embeddingArray = embeddings.arraySync()[0]; // Single vector
    return res.json({ embedding: embeddingArray });
  } catch (error) {
    console.error("Error generating embedding:", error);
    return res.status(500).json({ error: "Failed to generate embedding." });
  }
});


// app.post("/search", async (req, res) => {
//   const { text} = req.body;
//   if (!text || !items) {
//     return res.status(400).json({ error: "Missing text or items." });
//   }

//   try {
//     const queryEmbedding = await model.embed([text]);
//     const queryVec = queryEmbedding.arraySync()[0];

//  const cosineSimilarity = (vecA, vecB) => {
//   const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
//   const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
//   const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
//   if (magnitudeA === 0 || magnitudeB === 0) return 0;
//   return dotProduct / (magnitudeA * magnitudeB);
// };

//     const results = Data
//       .map((item) => ({
//         ...item,
//         score: cosineSimilarity(queryVec, item.embedding),
//       }))
//       .sort((a, b) => b.score - a.score);

//     return res.json({ results });
//   } catch (err) {
//     console.error(err);
//     return res.status(500).json({ error: "Failed to search." });
//   }
// });

app.listen(port, () => {
  console.log(`Semantic backend listening at http://localhost:${port}`);
});
