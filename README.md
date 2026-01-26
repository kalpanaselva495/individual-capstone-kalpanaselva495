[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/G8XTrpUY)
# Units 1-3 Capstone: End-to-End Machine Learning Project

## Overview

This is your first complete machine learning project from start to finish. You will apply **everything you've learned in Units 1-3** by executing the full data science process:

1. **Define a problem** worth solving with your chosen dataset
2. **Explore and clean** your data thoroughly (EDA)
3. **Engineer features** to improve model performance
4. **Build a regression model** to predict a numerical target
5. **Build a classification model** by binning your target into categories
6. **Evaluate and iterate** on your models
7. **Deploy both models** to a live Streamlit web application

This project will become part of your **professional portfolio on GitHub**. You'll also practice using Git like a professional developer.

---

## Timeline & Checkpoints

| Checkpoint | Due Date | What to Submit |
|------------|----------|----------------|
| **Checkpoint 1** | **Feb 1** | Problem statement defined, first-pass EDA started |
| **Checkpoint 2** | **Feb 8** | Complete EDA notebook, data cleaned, features engineered |
| **Checkpoint 3** | **Feb 15** | Regression model notebook complete, binning strategy submitted |
| **Checkpoint 4** | **Feb 22** | Classification model notebook complete |
| **Final Submission** | **Mar 1** | Streamlit app deployed to Streamlit Cloud |
| **Presentations** | **Mar 2-5** | Live demo in breakout groups |

### How to Submit

1. **Commit your work** to this GitHub repo by each deadline
2. **Submit your repo link** on the Canvas assignment page

---

## Git Workflow Expectations

Part of this capstone is learning to **use Git like a professional**. Your commit history matters!

### Commit Frequently
- Commit your work **regularly** as you make progress
- Don't wait until the deadline to make one giant commit
- A good rule: commit whenever you complete a logical piece of work

### Write Meaningful Commit Messages
Your commit messages should describe **what you did** clearly:

**Good commit messages:**
```
Add initial EDA with target distribution analysis
Handle missing values in bedroom and bathroom columns
Implement baseline linear regression model
Fix feature scaling issue in classification pipeline
Add user input form to Streamlit app
```

**Bad commit messages:**
```
update
fixed stuff
asdfasdf
final version
changes
```

### Example Commit Workflow
```bash
# After completing your problem statement
git add notebooks/01_problem_statement_and_eda.ipynb
git commit -m "Define problem statement and load dataset"

# After handling missing values
git add notebooks/01_problem_statement_and_eda.ipynb
git commit -m "Handle missing values using median imputation"

# After creating visualizations
git add notebooks/01_problem_statement_and_eda.ipynb
git commit -m "Add correlation heatmap and distribution plots"

# Push your changes to GitHub
git push
```

### Why This Matters
- Employers look at your GitHub commit history
- Good commits make it easy to track your progress
- If something breaks, you can go back to a working version
- It shows you understand professional development workflows

---

## Project Structure

```
your-capstone-project/
├── README.md                    # This file (update with your project details)
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── 01_problem_statement_and_eda.ipynb   # Checkpoints 1 & 2
│   ├── 02_regression_model.ipynb            # Checkpoint 3
│   └── 03_classification_model.ipynb        # Checkpoint 4
│
├── data/
│   ├── raw/                     # Your original dataset (CSV)
│   └── processed/               # Cleaned/transformed data
│
├── models/                      # Saved model files (.pkl)
│
├── app/
│   ├── app.py                   # Your Streamlit application
│   └── utils.py                 # Helper functions for the app
│
└── helpers/
    └── model_helpers.py         # Utilities for saving/loading models
```

---

## Detailed Requirements

### Checkpoint 1 (Due: Feb 1)

Complete the first sections of `notebooks/01_problem_statement_and_eda.ipynb`:

- [ ] **Problem Statement**: What are you trying to predict? Why does it matter?
- [ ] **Dataset Description**: What is your data? Where did it come from?
- [ ] **Target Variable**: Clearly identify your numerical target column
- [ ] **Initial EDA**: Load data, check shape, data types, first look at distributions

### Checkpoint 2 (Due: Feb 8)

Complete the remaining sections of `notebooks/01_problem_statement_and_eda.ipynb`:

- [ ] **Complete EDA**: Distributions, correlations, relationships with target
- [ ] **Data Cleaning**: Handle missing values, outliers, data quality issues
- [ ] **Feature Engineering**: Create new features, encode categoricals, scale numericals
- [ ] **Save Processed Data**: Export clean data to `data/processed/`

### Checkpoint 3 (Due: Feb 15)

Complete `notebooks/02_regression_model.ipynb`:

- [ ] **Baseline Model**: Build a simple model (e.g., Linear Regression)
- [ ] **Model Iteration**: Try at least 2-3 different models or configurations
- [ ] **Feature Importance**: Analyze which features matter most for predictions
- [ ] **Feature Selection**: Select your top features for the final model (aim for 4-8 features)
- [ ] **Evaluation**: Use appropriate metrics (R², RMSE, MAE)
- [ ] **Best Model Selection**: Choose and justify your best model
- [ ] **Save Model**: Export to `models/regression_model.pkl`

**Also due by Feb 15:** Submit your **binning strategy** to Abishek for approval (see Classification section below)

### Checkpoint 4 (Due: Feb 22)

Complete `notebooks/03_classification_model.ipynb`:

- [ ] **Create Binned Target**: Convert regression target to categories
- [ ] **Justify Binning**: Explain why your binning strategy makes sense
- [ ] **Baseline Model**: Build a simple classifier
- [ ] **Model Iteration**: Try at least 2-3 different models
- [ ] **Feature Selection**: Use the same selected features from your regression model (or justify different choices)
- [ ] **Evaluation**: Use appropriate metrics (accuracy, precision, recall, F1, confusion matrix)
- [ ] **Best Model Selection**: Choose and justify your best model
- [ ] **Save Model**: Export to `models/classification_model.pkl`

### Final Submission (Due: Mar 1)

Deploy your Streamlit app to Streamlit Cloud:

- [ ] **Working App**: App loads without errors
- [ ] **Sensible Input Form**: Only ask users for your selected features (4-8 inputs, not 20+)
- [ ] **Regression Predictions**: Users can input values and get a numerical prediction
- [ ] **Classification Predictions**: Users can input values and get a category prediction
- [ ] **Both Models Accessible**: Either on one page or separate pages/tabs
- [ ] **Deployed URL**: App is live on Streamlit Cloud
- [ ] **README Updated**: Your project details section is filled out (this is your portfolio!)

---

## Deploying to Streamlit Cloud

### Important: Moving from GitHub Classroom to Your Personal GitHub

Streamlit Cloud requires your repo to be in your **personal GitHub account** (not in the GitHub Classroom organization). Before deploying, you'll need to move your project.

### Step-by-Step Migration

**1. Create a new repository in your personal GitHub account**
- Go to [github.com/new](https://github.com/new)
- Name it something professional (e.g., `used-car-price-predictor` or `housing-price-ml`)
- Keep it **Public** (required for free Streamlit Cloud hosting)
- Do NOT initialize with README, .gitignore, or license (your repo already has these)

**2. Change your local repo's remote URL**
```bash
# Check your current remote
git remote -v

# Change to your new personal repo
git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-NEW-REPO-NAME.git

# Push all your code to your personal repo
git push -u origin main
```

**3. Verify your code is in your personal repo**
- Go to your new repo on GitHub
- Make sure all your files, notebooks, and commit history are there

**4. Deploy to Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select your personal repo
- Set main file path to: `app/app.py`
- Click "Deploy"

### Why This Matters

- Your **personal GitHub** is your professional portfolio
- Employers and recruiters look at your GitHub profile
- This project showcases your ML skills—make it visible!
- Keep this repo **public** so anyone can see your work

### Update Your README!

Before deploying, make sure to update the "Your Project Details" section at the bottom of this README. This is what visitors see when they land on your GitHub repo page. Make it look professional!

---

## Important Guidelines

### On Model Performance

**Your goal is NOT to achieve a specific R² or accuracy score.**

I'm evaluating your ability to:
- Build a reasonable baseline model
- Iterate and attempt improvements
- Demonstrate understanding of the modeling process
- Make thoughtful, justified decisions

Your model's performance is tied to how well you clean your data and engineer features. A "difficult" dataset with excellent process work will score better than an "easy" dataset with lazy execution.

### On Dataset Size

Your dataset must be committed to this repo. If your file is too large (>100MB):
- Reduce rows intelligently (don't just randomly delete)
- Maintain the distribution of your target variable
- Document what you removed and why

### On Feature Selection

**A key part of this project is identifying which features actually matter.**

Your dataset may have 15, 20, or even 50+ columns. Your job is to:
1. Analyze feature importance (correlation analysis, model-based importance scores)
2. Select the **4-8 most impactful features** for your final model
3. Justify why you chose these features

**Why this matters:**
- **For your model**: More features ≠ better predictions. Irrelevant features add noise.
- **For your Streamlit app**: Users shouldn't fill out 20 input fields to get one prediction. A good app asks for only the information that actually matters.
- **For your career**: In the real world, you'll often need to explain to stakeholders which variables drive predictions. This is a critical skill.

**How to analyze feature importance:**
- Correlation with target variable
- Feature importance from tree-based models (Random Forest, XGBoost)
- Coefficients from regularized models (Lasso)
- Domain knowledge (what *should* matter?)

### On Binning for Classification

By **Feb 15**, send your binning strategy to Abishek on Slack for approval. Include:
- How you're converting your regression target to categories
- Why this binning makes sense for your problem domain
- How many categories and what the thresholds are

**Example binning strategies:**
- Quartiles (25th, 50th, 75th percentiles)
- Domain-based (e.g., price ranges that make sense for housing)
- Equal-width bins
- Custom thresholds based on business meaning

---

## Grading Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Problem Statement** | 5% | Clear question, appropriate dataset, well-defined target |
| **EDA & Data Cleaning** | 15% | Thorough exploration, visualizations, insights, proper cleaning |
| **Feature Engineering** | 10% | Thoughtful transformations, justified decisions |
| **Regression Model** | 20% | Baseline built, improvements attempted, process demonstrated |
| **Classification Model** | 20% | Justified binning, baseline + improvements, process demonstrated |
| **Streamlit Deployment** | 20% | Working app, user input → predictions, both models functional |
| **Code Quality & Docs** | 5% | Clean code, comments, organized repo |
| **Presentation** | 5% | Clear demo, explains decisions, answers questions |

**Late submissions will be penalized.** If you anticipate issues meeting a deadline, reach out to Abishek as early as possible.

---

## Presentations (Mar 2-5)

You will present your project to your breakout group:

1. **Live Demo** (primary focus): Show your deployed Streamlit app working
2. **Brief Walkthrough**: Explain your process and key decisions
3. **Q&A**: Answer questions from classmates

After presentations, you'll submit feedback on your classmates' projects via a form. Abishek will compile anonymous feedback and share it with each student.

---

## Default Dataset (If You Didn't Find Your Own)

If you didn't find your own dataset, you'll be using the **Craigslist Cars and Trucks Dataset**:

**Dataset Link:** https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

**Your task:** Predict the **price** of used vehicles.

### Why This Dataset?

This dataset is notoriously messy and will require significant data cleaning work. You'll encounter:

- **Extensive missing values** across many columns
- **Invalid/nonsensical entries** (e.g., odometer readings like "1234567")
- **Outliers galore** (prices of $1 or $999,999,999)
- **Inconsistent categorical data** that needs standardization
- **A large file** that you'll need to reduce intelligently

This is what real-world data looks like. Consider it good practice for your future career.

### Getting Started with This Dataset

1. Download the dataset from Kaggle (you'll need a free Kaggle account)
2. The file is large (~1.4GB) - you'll need to sample it down to <100MB for GitHub
3. When reducing the dataset:
   - Keep the distribution of prices representative
   - Document your sampling strategy in your notebook
   - A sample of 50,000-100,000 rows should be sufficient

### Suggested Approach

Given the messiness, plan to spend significant time on Checkpoint 2 (EDA & Cleaning). You'll likely need to:
- Drop columns with too many missing values
- Handle price outliers (filter unrealistic prices)
- Clean the odometer column
- Decide which categorical features to keep/encode
- Consider geographic features (state/region)

**Important:** This dataset has 25+ columns. You will NOT use all of them. A major part of your job is figuring out which 4-8 features actually predict price well. Think about it: if you're building a "used car price estimator" app, what information would a user realistically have and be willing to enter?

---

## Getting Started

### 1. Clone this repo
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
- Place your CSV file in `data/raw/`
- Update the data loading code in your notebooks

### 5. Start working!
- Open `notebooks/01_problem_statement_and_eda.ipynb`
- Follow the guided structure in each notebook

---

## Useful Commands

### Git basics
```bash
git add .                    # Stage all changes
git commit -m "Your message" # Commit with a message
git push                     # Push to GitHub
```

### Run Streamlit locally
```bash
streamlit run app/app.py
```

---

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## Questions?

Reach out to Abishek on Slack if you have any questions or get stuck!

---

## Your Project Details (Update This Section!)

> **Important:** This section is the first thing visitors see when they land on your GitHub repo. This is your portfolio—make it professional and complete! Update this before your final submission.

**Student Name:** [Your Name]

**Dataset:** [Dataset name and source - include a link if from Kaggle]

**Problem Statement:** [What are you trying to predict and why? Write 2-3 sentences explaining the value of this prediction.]

**Target Variable:** [Column name - e.g., "price" or "salary"]

**Selected Features:** [List the 4-8 features your final model uses]

**Best Regression Model:** [Model type and key metric - e.g., "Random Forest (R² = 0.82)"]

**Best Classification Model:** [Model type and key metric - e.g., "Gradient Boosting (Accuracy = 85%)"]

**Deployed App URL:** [Add your Streamlit Cloud URL once deployed]

### Project Highlights

[Write 2-3 bullet points about interesting findings or challenges you overcame. This helps employers understand your thought process!]

-
-
-
