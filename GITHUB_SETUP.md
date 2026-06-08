# Publishing to GitHub — Step-by-Step

## 1. Create the repo on GitHub

Go to https://github.com/new and set:
- **Repository name:** `roofline-model-llm-inference`
- **Description:** `Why more compute doesn't mean faster AI — the Roofline Model explained, with an interactive calculator and Python workload analyzer.`
- **Visibility:** Public
- **Do NOT** initialize with README (you already have one)

Click **Create repository**.

## 2. Push from your local machine

Open Terminal and run:

```bash
cd "path/to/roofline-model-llm-inference"

git init
git add .
git commit -m "Initial publish: Roofline Model article, calculator, and Python analyzer"

git remote add origin https://github.com/YOUR_USERNAME/roofline-model-llm-inference.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub handle.

## 3. Add topics (for discoverability)

After pushing, go to your repo page → click the ⚙️ gear next to "About" → add these topics:

```
llm  inference  gpu  tpu  machine-learning  ai-infrastructure  hardware  roofline-model  compute-efficiency  deep-learning
```

## 4. Enable GitHub Pages (for the calculator)

Go to **Settings → Pages**:
- Source: `Deploy from a branch`
- Branch: `main`, folder: `/ (root)`
- Click **Save**

Your calculator will be live at:
`https://YOUR_USERNAME.github.io/roofline-model-llm-inference/calculator/`

Update the README badge URL once Pages is live.

## 5. Update the README

Replace `yourusername` with your actual GitHub username in:
- The badge URL at the top of `README.md`
- The script example URL in the "Try It Yourself" section

## 6. Share it

Suggested cross-post caption for LinkedIn / X / Substack:

> Just published Part 1 of my series on AI compute efficiency on GitHub — "Why More Compute Does Not Mean Faster AI."
>
> Most LLM inference runs at ~0.2% of chip potential. I built an interactive Roofline calculator and a Python tool to check where your workload actually sits.
>
> ⭐ the repo if it's useful: github.com/YOUR_USERNAME/roofline-model-llm-inference
