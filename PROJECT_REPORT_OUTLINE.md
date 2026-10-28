# Project report outline

1. **Title page** — GeoGen AI: Satellite Image Generation and Data Augmentation Using GANs.
2. **Abstract** — State the objective, dataset, GAN approach, and measured outcomes after experiments. Do not prefill results.
3. **Introduction** — Explain land use classification and the difficulty of obtaining balanced remote-sensing data.
4. **Problem statement and objectives** — Generate class-conditioned EuroSAT-like RGB tiles and investigate augmentation.
5. **Related work** — Summarize EuroSAT and conditional GAN literature with citations.
6. **Dataset** — 27,000 labeled RGB images, ten classes, 64×64 tiles; document source, license, split and preprocessing.
7. **Methodology** — Explain generator, projection discriminator, adversarial loss, optimizer and training settings.
8. **Implementation** — Python, PyTorch, Google Colab GPU, checkpointing and Streamlit inference.
9. **Experiments** — Record hardware/runtime, seeds, epochs, batch size, classifier baseline and augmentation setup.
10. **Results** — Insert only actual loss curves, generated samples, classifier metrics, and limitations.
11. **Discussion and limitations** — Image fidelity, mode collapse, class confusion, geographic/domain bias, synthetic-data risks.
12. **Conclusion and future work** — Summarize findings and propose multispectral data, improved GANs, and stronger evaluation.
13. **References and appendices** — EuroSAT citation, license terms, source code and viva answers.

