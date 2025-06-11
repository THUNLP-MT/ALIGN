# ALIGN: Agent‑Environment Alignment via Automated Interface Generation

![ALIGN Framework](./assets/system.png)

---

## 🔍 What is ALIGN?

Large‑language‑model (LLM) agents often fail because their *interfaces* (the action & observation layer that sits between the agent and the environment) neglect to reveal hidden pre‑conditions or constraints.  **ALIGN** is a system that **automatically generates a richer, better‑aligned interface** without touching either the agent logic *or* the environment code.

<!-- <div align="center">
<img src="./assets/align_misalignment_vs_align.png" width="600" alt="Illustration of misalignment vs. ALIGN‑generated interface"/>
</div> -->
![ALIGN Misalignment vs. ALIGN Interface](./assets/align_misalignment_vs_align.png)

### Key Contributions

Our key contributions can be summarized as follows:
- We identify and characterize the *agent-environment misalignmeent* problem, empirically demonstrating its prevalence across diverse domains and its role as a siggnificant bottleneck to agent performance.
- We introduce *ALIGN*, the first framework automatically generates alligned interfaces to alleviate agent-environment misalignment, without modifying agent logic orenvironment code.
- We demonstrate the effectiveness and generalizability of *ALIGN* across three domains, with up to a 45.67% success rate improvement on ALFWorld and consistent boosts on ScienceWorld, WebShop, and M3ToolEval.

---

## ✨ Results

![ALIGN Main Results](./assets/main_results.png)

---

## ✏️ Citing ALIGN

If you find this work useful, please cite:

```bibtex
@misc{liu2025agentenvironmentalignmentautomatedinterface,
      title={Agent-Environment Alignment via Automated Interface Generation}, 
      author={Kaiming Liu and Xuanyu Lei and Ziyue Wang and Peng Li and Yang Liu},
      year={2025},
      eprint={2505.21055},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.21055}, 
}
```

---

## 💬 Contact & Questions

Open an issue or reach us at `chatkming@gmail.com`.

Enjoy and happy aligning! 🎉
