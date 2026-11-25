# Errors_of_Intelligence_Mistakes_in_ResNet50

An attempt to ask whether a wrong output of an artificial vision system can be a *mistake* and not merely an *error*, using Biological Mistake Theory (BMT). The test case is the false "knife" answers of ResNet-50 on edge, silhouette and cue-conflict images. The work builds on Geirhos et al. (2019, 2020) and Tartaglini et al. (2022), and on the stimuli and EMCE framework of Darshini (2026), to study how a wrong answer behaves when the input is changed. Each wrong answer is placed on an avoidability ladder with four rungs: resistance, escape or reversion, unstable correction, stable correction.


## How to use
ResNet-50 is used frozen with standard ImageNet weights; no weights are changed in any experiment. The 1000 ImageNet outputs are mapped to the 16 cue-conflict categories. The stimuli (edge, silhouette and cue-conflict images) are from Darshini (2026), drawn from Geirhos et al. (2019, 2020): https://github.com/Darshini-CogSci/AI_Mistakes

## Code structure
All notebooks were developed and run in Google Colab.

1. `Replication_1.ipynb`: replication of the EMCE pipeline and of the error-consistency analysis (Geirhos et al., 2020). **[check]**
2. `Replication_2_+_GradCam.ipynb`: second replication step, plus Grad-CAM at the last block of layer 4, comparing the share of activation on the diagnostic region (contour, silhouette or foreground) with a random region. **[check]**
3. `Probe.ipynb`: balanced logistic-regression linear probes on activations at Layer 2, Layer 3 and AvgPool, read out on ResNet-50's own errors, to test whether the correct category is present inside the network.
4. `Cue_Conflict_intervention.ipynb`: texture blending on the 47 false-knife cue-conflict images. The object is fixed and a fraction α of the background texture (1.00, 0.75, 0.50, 0.25, 0.00) is kept, and each image is classed on the avoidability ladder.
5. `Edge_Silhouette_intervention.ipynb` *(being added)*: interventions on edge and silhouette images: edge degradation across four edge versions, 32 × 32 patch occlusion with a permutation test, and silhouette unmasking (restoring real edge patches to the 51 bare silhouettes). **[check filename]**

Not in this repository:
- The ChatGPT comparison was run manually (fixed prompts, run sheets), so there is no notebook for it.
- The designed but unexecuted tests (Stylized-ImageNet reference model, cue 2 × 2, direction ablation, predictive coding network) are described in the report only.

For the stimuli, human data and the original manipulations, please see the works in the References section.

## Installation & Usage

1. Clone the repository:
git clone https://github.com/singhsaracis/Error-and-Mistakes-in-AI-Sara.git
2. Open the notebook you want in Google Colab (upload it, or File → Open notebook → GitHub). Select a GPU runtime (Runtime → Change runtime type).
3. Update the data paths in the first cells to point to your copy of the stimuli. Colab already includes PyTorch, NumPy and scikit-learn. Install any extras with the `pip install` line in the first cell of each notebook (e.g. a Grad-CAM package). **[check]**
4. Run the notebooks in the order above.

## Data
The image sets are the edge, silhouette and cue-conflict stimuli of - https://github.com/Darshini-CogSci/AI_Mistakes_Error_Framework

## Note on AI assistance
Some of the code was generated with Claude AI and then reviewed and edited by the author.

## References
* Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *ICLR*. https://doi.org/10.48550/arxiv.1811.12231
* Geirhos, R., Meding, K., & Wichmann, F. A. (2020). Beyond accuracy: Quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency. *NeurIPS, 33*.
* Tartaglini, A. R., Vong, W. K., & Lake, B. M. (2022). A Developmentally-Inspired Examination of Shape versus Texture Bias in Machines. *Proceedings of the Annual Meeting of the Cognitive Science Society, 44*. https://doi.org/10.48550/arxiv.2202.08340
* Darshini, N. (2026). All errors are not equal: An efficiency-modulated cumulative error framework. Major project report, IIT Delhi.
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
* Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
* Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644.
* Firestone, C. (2020). Performance vs. competence in human–machine comparisons. *PNAS, 117*(43). https://doi.org/10.1073/pnas.1905334117
* Oderberg, D. S., Hill, J., Austin, C., Bojak, I., Cinotti, F., & Gibbins, J. M. (2025). Biological mistake theory and the question of function. *Philosophy of Science, 92*(2). https://doi.org/10.1017/psa.2024.56
