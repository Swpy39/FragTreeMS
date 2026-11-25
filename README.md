<h1 align="center">FragTreeMS</h1>

<p align="center">
  <strong><em>Mechanism-grounded fragmentation trees enable explicit chemical reasoning for electron ionization mass spectrum prediction</em></strong>
</p>

<p align="center"><sub>Last updated: 2026.11.25</sub></p>
<p align="center">
  <a href="#overview">🧭 Overview</a> &nbsp;·&nbsp;
  <a href="#chs-benchmark-results">📊 CHS results</a> &nbsp;·&nbsp;
  <a href="#method-workflow">⚙️ Method</a> &nbsp;·&nbsp;
  <a href="#visualization">🔬 Interactive demo</a>
</p>

<a id="overview"></a>
## 🧭 Overview

FragTreeMS makes EI ion formation explicit as a source-mechanism-product fragmentation tree. Candidate-guided repair refines the tree, and each complete pathway conditions intensity prediction through Path Cross-Attention and Inter-Peak Interaction.

<p align="center">
  <a href="./Images/Main.pdf">
    <img src="./Images/Main.png"
         alt="FragTreeMS framework: fragmentation-tree generation and repair followed by pathway-conditioned intensity prediction"
         width="80%">
  </a>
</p>

<p align="center">
  <strong>Figure 1 | Overview of the FragTreeMS framework.</strong>
</p>

<a id="chs-example"></a>
## 🧪 CHS example

A representative sulfur-containing molecule, its EI spectrum and its mechanism-coloured fragmentation tree are shown below.

<p align="center">
  <a href="./Images/Figures-S1.pdf">
    <img src="./Images/Figures-S1.svg"
         alt="Representative CHS molecule, EI spectrum, fragmentation tree and mechanism legend"
         width="60%">
  </a>
</p>

<p align="center">
  <strong>Figure S1 | A representative CHS molecule and its EI spectrum, mechanism-coloured fragmentation tree and assigned ion pathways.</strong>
</p>

<a id="chs-benchmark-results"></a>
## 📊 CHS benchmark results

The tables compare direct and triplet-input settings, baseline predictors and full FragTreeMS models.

### 📈 Spectrum-prediction metrics

<p><strong>Direct versus triplet-input settings.</strong> Base-peak accuracy and cosine similarity are shown side by side for each backbone.</p>

<div align="center">
<table align="center" width="100%" cellpadding="3" cellspacing="0">
<thead>
<tr>
<th align="center" nowrap><small>System&nbsp;/&nbsp;model</small></th>
<th align="center" nowrap><small>Direct&nbsp;·&nbsp;base&#8209;peak</small></th>
<th align="center" nowrap><small>Direct&nbsp;·&nbsp;cosine</small></th>
<th align="center" nowrap><small>Triplets&nbsp;·&nbsp;base&#8209;peak</small></th>
<th align="center" nowrap><small>Triplets&nbsp;·&nbsp;cosine</small></th>
</tr>
</thead>
<tbody>
<tr><td align="center" nowrap><small>GPT&nbsp;·&nbsp;GPT&#8209;5.5</small></td><td align="center" nowrap><small>0.320261&nbsp;±&nbsp;0.019608</small></td><td align="center" nowrap><small>0.580188&nbsp;±&nbsp;0.011335</small></td><td align="center" nowrap><small>0.278867&nbsp;±&nbsp;0.003774</small></td><td align="center" nowrap><small>0.554586&nbsp;±&nbsp;0.001029</small></td></tr>
<tr><td align="center" nowrap><small>DeepSeek&nbsp;·&nbsp;DeepSeek&#8209;V4&#8209;Pro</small></td><td align="center" nowrap><small>0.455338&nbsp;±&nbsp;0.021010</small></td><td align="center" nowrap><small>0.652453&nbsp;±&nbsp;0.003365</small></td><td align="center" nowrap><small>0.411765&nbsp;±&nbsp;0.006536</small></td><td align="center" nowrap><small>0.622592&nbsp;±&nbsp;0.004737</small></td></tr>
<tr><td align="center" nowrap><small>Local&nbsp;·&nbsp;Qwen3&#8209;8B</small></td><td align="center" nowrap><small>0.039216&nbsp;±&nbsp;0.006536</small></td><td align="center" nowrap><small>0.123006&nbsp;±&nbsp;0.004331</small></td><td align="center" nowrap><small>0.209150&nbsp;±&nbsp;0.013072</small></td><td align="center" nowrap><small>0.190229&nbsp;±&nbsp;0.001188</small></td></tr>
<tr><td align="center" nowrap><small>Local&nbsp;·&nbsp;Llama3.1&#8209;8B</small></td><td align="center" nowrap><small>0.023965&nbsp;±&nbsp;0.013606</small></td><td align="center" nowrap><small>0.105070&nbsp;±&nbsp;0.004642</small></td><td align="center" nowrap><small>0.067538&nbsp;±&nbsp;0.007547</small></td><td align="center" nowrap><small>0.153230&nbsp;±&nbsp;0.005393</small></td></tr>
<tr><td align="center" nowrap><small>Local&nbsp;·&nbsp;Qwen2.5&#8209;14B</small></td><td align="center" nowrap><small>0.028322&nbsp;±&nbsp;0.003774</small></td><td align="center" nowrap><small>0.122590&nbsp;±&nbsp;0.005339</small></td><td align="center" nowrap><small>0.034858&nbsp;±&nbsp;0.007547</small></td><td align="center" nowrap><small>0.113549&nbsp;±&nbsp;0.006863</small></td></tr>
</tbody>
</table>
</div>

<p><strong>Baselines and full-model results.</strong></p>

<div align="center">
<table align="center" width="100%" cellpadding="3" cellspacing="0">
<thead>
<tr>
<th align="center" nowrap><small>System</small></th>
<th align="center" nowrap><small>Model&nbsp;/&nbsp;setting</small></th>
<th align="center" nowrap><small>Base&#8209;peak&nbsp;accuracy</small></th>
<th align="center" nowrap><small>Cosine</small></th>
</tr>
</thead>
<tbody>
<tr><td align="center" nowrap><small>Baseline</small></td><td align="center" nowrap><small>NEIMS</small></td><td align="center" nowrap><small>0.457516&nbsp;±&nbsp;0.017292</small></td><td align="center" nowrap><small>0.737847&nbsp;±&nbsp;0.001619</small></td></tr>
<tr><td align="center" nowrap><small>Baseline</small></td><td align="center" nowrap><small>CFM&#8209;ID</small></td><td align="center" nowrap><small>0.294118&nbsp;±&nbsp;0.000000</small></td><td align="center" nowrap><small>0.528892&nbsp;±&nbsp;0.000000</small></td></tr>
<tr><td align="center" nowrap><small>FragTreeMS</small></td><td align="center" nowrap><small>Qwen3&#8209;8B&nbsp;·&nbsp;Full&nbsp;model</small></td><td align="center" nowrap><small>0.490196&nbsp;±&nbsp;0.017292</small></td><td align="center" nowrap><small>0.735000&nbsp;±&nbsp;0.003529</small></td></tr>
<tr><td align="center" nowrap><small>FragTreeMS</small></td><td align="center" nowrap><small>Llama3.1&#8209;8B&nbsp;·&nbsp;Full&nbsp;model</small></td><td align="center" nowrap><small>0.450980&nbsp;±&nbsp;0.006536</small></td><td align="center" nowrap><small>0.724851&nbsp;±&nbsp;0.002479</small></td></tr>
<tr><td align="center" nowrap><small>FragTreeMS</small></td><td align="center" nowrap><small>Qwen2.5&#8209;14B&nbsp;·&nbsp;Full&nbsp;model</small></td><td align="center" nowrap><small>0.448802&nbsp;±&nbsp;0.024705</small></td><td align="center" nowrap><small>0.729116&nbsp;±&nbsp;0.006065</small></td></tr>
</tbody>
</table>
</div>

<a id="cho-benchmark-results"></a>
## 🍃 CHO benchmark results

Base-peak accuracy and cosine similarity on the CHO test set.

<div align="center">
<table align="center" width="100%" cellpadding="3" cellspacing="0">
<thead>
<tr>
<th align="center" nowrap><small>System</small></th>
<th align="center" nowrap><small>Model&nbsp;/&nbsp;setting</small></th>
<th align="center" nowrap><small>Base&#8209;peak&nbsp;accuracy</small></th>
<th align="center" nowrap><small>Cosine</small></th>
</tr>
</thead>
<tbody>
<tr><td align="center" nowrap><small>Local</small></td><td align="center" nowrap><small>Qwen3&#8209;8B&nbsp;·&nbsp;Direct</small></td><td align="center" nowrap><small>0.046296&nbsp;±&nbsp;0.008019</small></td><td align="center" nowrap><small>0.243866&nbsp;±&nbsp;0.005813</small></td></tr>
<tr><td align="center" nowrap><small>Local</small></td><td align="center" nowrap><small>Qwen3&#8209;8B&nbsp;·&nbsp;Triplets</small></td><td align="center" nowrap><small>0.032407&nbsp;±&nbsp;0.008019</small></td><td align="center" nowrap><small>0.175223&nbsp;±&nbsp;0.012039</small></td></tr>
<tr><td align="center" nowrap><small>Local</small></td><td align="center" nowrap><small>Llama3.1&#8209;8B&nbsp;·&nbsp;Direct</small></td><td align="center" nowrap><small>0.120370&nbsp;±&nbsp;0.016038</small></td><td align="center" nowrap><small>0.314998&nbsp;±&nbsp;0.001243</small></td></tr>
<tr><td align="center" nowrap><small>Local</small></td><td align="center" nowrap><small>Llama3.1&#8209;8B&nbsp;·&nbsp;Triplets</small></td><td align="center" nowrap><small>0.060185&nbsp;±&nbsp;0.008019</small></td><td align="center" nowrap><small>0.196809&nbsp;±&nbsp;0.013111</small></td></tr>
<tr><td align="center" nowrap><small>Local</small></td><td align="center" nowrap><small>Qwen2.5&#8209;14B&nbsp;·&nbsp;Direct</small></td><td align="center" nowrap><small>0.115741&nbsp;±&nbsp;0.021216</small></td><td align="center" nowrap><small>0.275989&nbsp;±&nbsp;0.006050</small></td></tr>
<tr><td align="center" nowrap><small>Local</small></td><td align="center" nowrap><small>Qwen2.5&#8209;14B&nbsp;·&nbsp;Triplets</small></td><td align="center" nowrap><small>0.069444&nbsp;±&nbsp;0.000000</small></td><td align="center" nowrap><small>0.253684&nbsp;±&nbsp;0.015794</small></td></tr>
<tr><td align="center" nowrap><small>Baseline</small></td><td align="center" nowrap><small>NEIMS</small></td><td align="center" nowrap><small>0.430556&nbsp;±&nbsp;0.036747</small></td><td align="center" nowrap><small>0.710813&nbsp;±&nbsp;0.002057</small></td></tr>
<tr><td align="center" nowrap><small>Baseline</small></td><td align="center" nowrap><small>CFM&#8209;ID</small></td><td align="center" nowrap><small>0.388889&nbsp;±&nbsp;0.000000</small></td><td align="center" nowrap><small>0.616766&nbsp;±&nbsp;0.000000</small></td></tr>
<tr><td align="center" nowrap><small>FragTreeMS</small></td><td align="center" nowrap><small>Qwen3&#8209;8B&nbsp;·&nbsp;Full&nbsp;model</small></td><td align="center" nowrap><small>0.407407&nbsp;±&nbsp;0.008019</small></td><td align="center" nowrap><small>0.701937&nbsp;±&nbsp;0.001249</small></td></tr>
<tr><td align="center" nowrap><small>FragTreeMS</small></td><td align="center" nowrap><small>Llama3.1&#8209;8B&nbsp;·&nbsp;Full&nbsp;model</small></td><td align="center" nowrap><small>0.449074&nbsp;±&nbsp;0.008019</small></td><td align="center" nowrap><small>0.705430&nbsp;±&nbsp;0.004012</small></td></tr>
<tr><td align="center" nowrap><small>FragTreeMS</small></td><td align="center" nowrap><small>Qwen2.5&#8209;14B&nbsp;·&nbsp;Full&nbsp;model</small></td><td align="center" nowrap><small>0.421296&nbsp;±&nbsp;0.016038</small></td><td align="center" nowrap><small>0.701659&nbsp;±&nbsp;0.002526</small></td></tr>
</tbody>
</table>
</div>

<a id="method-workflow"></a>
## ⚙️ Method workflow

1. **Annotation curation.** Initial mechanism annotations are generated with GPT-4.1, refined in a second GPT-4.1 pass, completed and corrected with GPT-5, and then reviewed manually. The released annotations are therefore curated records rather than unreviewed model outputs.
2. **Fragmentation tree reasoning and repair.** A language model first generates a fragmentation tree containing the predicted ion inventory, structural fragments, relations between precursor and product ions and mechanism labels. Diagnostic ion candidates then guide a self-review pass that recovers plausible omissions while preserving the explicit tree structure.
3. **Intensity prediction.** The repaired tree and its complete ion formation pathways are encoded to predict normalized EI peak intensities.

## 📂 Repository structure

```text
FragTreeMS/
├── annotation_pipeline/
│   ├── generation/
│   ├── refinement/
│   └── finalization/
├── stage1_tree_generation/
├── stage2_intensity_prediction/
├── data/
│   ├── acyclic_alkane_data/
│   ├── alkene_data/
│   ├── cycloalkane_data/
│   └── aromatic_hydrocarbon_data/
├── Images/
├── README.md
└── requirements.txt
```

| Directory | Description |
| --- | --- |
| `annotation_pipeline/` | Scripts for initial annotation generation, refinement and final correction. |
| `stage1_tree_generation/` | Training and inference code for fragmentation tree generation and self-review repair. |
| `stage2_intensity_prediction/` | Training and inference code for pathway-conditioned intensity prediction. |
| `data/` | Example of annotated data format. |
| `Images/` | Images for this project. |

## 🔎 Current scope

The mechanism vocabulary included in this release spans the principal ion formation and fragmentation processes commonly observed in EI mass spectrometry. The main validation datasets focus on 70 eV electron ionization (EI) mass spectra from four hydrocarbon classes: acyclic alkanes, alkenes, cycloalkanes and aromatic hydrocarbons. A supplementary CHS evaluation extends the reported benchmark to sulfur-containing molecules; its results and a worked example are presented at the top of this README.

The tree reasoning experiments use Qwen3-8B, Llama3.1-8B-Instruct and Qwen2.5-14B-Instruct as alternative backbone models. The intensity model uses the same pathway representation and evaluation protocol across backbones.

<a id="visualization"></a>
## 🔬 Visualization

FragTreeMS provides molecule-level visualizations of the inferred fragmentation tree, repaired product ions, complete formation pathways and predicted EI spectrum. The interactive demonstration is available at [https://pec.cup.edu.cn/FragTreeMS](https://pec.cup.edu.cn/FragTreeMS).

![Conceptual overview of FragTreeMS](./Images/visualization.png)
