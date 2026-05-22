# Student Repository for the AI Engineering Course

This repository is created from a template and is used **simultaneously** for:

- completing homework assignments for seminars;
- developing and presenting the final project.

Please do not change the base directory structure (`homeworks/` and `project/`) to make it easier for instructors to review the work.

---

## Student Info

Fill in this block **before starting work**:

- Full Name: `Сапармурадов Сапармурат`
- Group: `БФБО-01-24`
- Contact: `@benotlikethose5919`
- E-mail: `saparmurat.saparmuradov@mail.ru`

---

## Repository Structure

The repository has two main areas:

- `homeworks/` — seminar homework assignments;
- `project/` — final mini-project.

Details:

- `homeworks/`
  - `README.md` — brief homework formatting guidelines.
  - `HW01/`, `HW02/`, `HW03/`, ... — separate folder for each assignment.
    Each assignment must have **one main file** with the same number, e.g.:
    - `homeworks/HW01/HW01.ipynb`
    - `homeworks/HW02/HW02.ipynb`
    - etc.

- `project/`
  - `README.md` — project passport and launch instructions.
  - `requirements.txt` — dependencies **for the project only**.
  - `notebooks/` — experimental notebooks, EDA, prototypes.
  - `src/` — main project code (modules, pipelines, services).
  - `data/` — demo/training data (no personal or confidential data).
  - `configs/` — configuration files, `.env.example` templates, etc.
  - `tests/` — tests (if used).
  - `artifacts/` — saved models, reports, training artifacts.

---

## How to Work with Homework

1. For each new assignment, create a folder:
   - `homeworks/HW01/`
   - `homeworks/HW02/`
   - `homeworks/HW03/`
   - etc.

2. The main file for the assignment must be named:
   - `HW01.ipynb` inside `HW01/`;
   - `HW02.ipynb` inside `HW02/`;
   - etc.

3. If needed, you may add extra files to the corresponding folder (`.py`, additional notebooks, auxiliary data), as long as it does not conflict with the assignment requirements.

4. **Do not rename** the `homeworks/` folder or `HWNN/` folders (e.g., `HW01/`, `HW02/`, etc.) to avoid breaking automated and visual checks.

---

## How to Work on the Project

- Place all project code and materials **only** inside the `project/` folder.
- In `project/README.md`, describe:
  - the project goal;
  - a brief idea;
  - how to run the project (commands, dependencies, parameters);
  - how to reproduce the demo for the defense.
- Maintain a separate report in `project/report.md`:
  - problem statement and metrics;
  - data and experiment description;
  - model comparison and final model selection.
- Use the checklist in `project/self-checklist.md` for self-check before submission.
- Pin all project dependencies in `project/requirements.txt`.

---

## Final Project Grading

The final project is graded on a 5-point scale (2–5) based on:

- meeting the minimum requirements for service functionality and project structure;
- the number of completed checklist items in `project/self-checklist.md`;
- the quality of the report in `project/report.md` and overall implementation neatness.

Approximate guidelines:

- if the project does not meet minimum requirements (won't run, missing key functionality, gross violations, obvious plagiarism) — grade **2**;
- if the minimum is met but **fewer than 5** checklist items are completed — grade **3**;
- if **at least 5** checklist items are completed — grade **4**;
- if **at least 9** checklist items are completed — grade **5**.

The final decision rests with the instructor and may take into account additional project strengths and deadline adherence.

---

## Security and Data Handling

- Do not commit passwords, tokens, keys, files with real personal data, or closed datasets to the repository.
- If the project needs data, use:
  - open datasets;
  - synthetic data;
  - anonymized samples.

Detailed rules and restrictions are described in [`SECURITY.md`](./SECURITY.md). Please read it before starting work.

---
