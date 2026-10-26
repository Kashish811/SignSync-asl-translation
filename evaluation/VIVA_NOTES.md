# SignSync – what you must be able to explain (viva, 7 marks)
1. Task: continuous ASL video -> English sentence (How2Sign sentence-level clips, official re-aligned translations).
2. Pipeline: video -> RTMPose whole-body keypoints (133 pts) -> normalise (crop/scale by body) -> 4 part graphs
   (body 9 nodes, each hand 21 nodes, face subset 18 nodes) -> per-part ST-GCN (spatial graph conv over joints + temporal conv over frames)
   -> temporal fusion GCN -> mean over joints -> concat 4 parts (+ learned part bias) -> Linear(1024->768) -> prefix tokens
   "Translate sign language video to English:" + pose tokens -> mT5-base encoder-decoder -> beam search (4 beams).
3. Why a GNN: skeleton = graph (joints=nodes, bones=edges). Graph conv aggregates a joint with its neighbours (A_hat X W);
   temporal conv aggregates across frames. Separate graphs per part because hands carry handshape detail and need their own resolution.
4. Why a pretrained text decoder: How2Sign alone (~35k sentences) is too small to learn English; mT5 brings language knowledge.
5. What is borrowed (say it first!): architecture + pretrained weights + pose extraction from Uni-Sign (CC BY-NC 4.0).
   What is yours: data pipeline from official How2Sign files, evaluation harness, GNN-part ablations, shuffled-input control,
   error analysis, demo.
6. Metrics: BLEU-n (n-gram precision with brevity penalty), ROUGE-L (longest common subsequence F1), optional BLEURT.
   WER is not standard for translation (insertions can push it >100%).
7. Controls: shuffled pose -> score collapses if the model really uses the input. Drop-part -> shows each graph's contribution.
8. Limits to state honestly: single-signer-set dataset (studio, frontal); scores are far below text-translation quality; BLEU-4 ~ low-teens
   is state of the art for this dataset; model can hallucinate fluent but wrong sentences.
