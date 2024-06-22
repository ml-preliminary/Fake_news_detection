#!/bin/bash

## 删除本地Git仓库
#rm -rf .git

# 重新初始化Git仓库
git init

# 设置git仓库的远程地址
git remote add origin git@github.com:ml-preliminary/Fake_news_detection.git

# 查找并将大于1MB的文件添加到.gitignore中
find . -type f -size +1M -exec echo {} >> .gitignore \;

# 添加所有的 .py 文件和 .sh 文件
find . -name "*.py" -o -name "*.sh" | xargs git add

# 提交更改
git commit -m "Add all .py and .sh files, ignore files larger than 1MB"

# 推送更改到远程仓库
git push -u origin master
