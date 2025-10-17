# 0. Create a new html version w/ code
jupyter nbconvert --to html --no-input Ceuas_stat.ipynb
# git rm -r --cached .

# 1. Stage your changes
git add .

# 2. Commit your changes
git commit -m "code update"

# 3. Push the commits to GitHub
git push origin main
