# 使用 Git Token 推送到 GitHub 的步骤

本文记录了在本地终端为当前项目配置 GitHub 访问令牌（Personal Access Token, PAT）并完成推送的完整流程，同时解释了常见报错 `not a git repository (or any of the parent directories): .git` 的原因与解决方法。

## 1. 进入项目根目录

在执行任何 Git 命令之前，请先切换到项目的 Git 仓库根目录。仓库根目录下应该存在一个隐藏文件夹 `.git`。例如：

```bash
cd /path/to/your/local/repo
```

若在错误的目录执行 `git remote add` 等命令，就会出现：

```
fatal: not a git repository (or any of the parent directories): .git
```

此时说明当前目录不是 Git 仓库，请重新切换到包含 `.git` 的项目根目录后再执行后续步骤。

## 2. 配置 Git 用户身份（首次使用时需要）

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "与你GitHub账户绑定的邮箱"
```

## 3. 设置环境变量保存 Token（推荐）

将你的 GitHub PAT 写入临时环境变量，避免直接暴露在命令历史中：

```bash
export GITHUB_TOKEN=github_pat_xxxxxxxxx
```

推送完成后可使用 `unset GITHUB_TOKEN` 清除环境变量。

## 4. 配置远程地址

两种方案二选一：

### 方案 A：覆盖默认远程 `origin`

```bash
git remote set-url origin https://${GITHUB_TOKEN}@github.com/<你的用户名>/<仓库名>.git
```

### 方案 B：新增专用推送远程（推荐）

```bash
git remote add push-origin https://${GITHUB_TOKEN}@github.com/<你的用户名>/<仓库名>.git
```

推送时使用：

```bash
git push push-origin <分支名>
```

## 5. 执行推送

### 5.1 确认本地分支和提交

推送之前先确认本地仓库中已经存在至少一次提交，并且当前分支名称正确：

```bash
git status
git branch
```

如果 `git status` 提示 “nothing to commit, working tree clean”，但 `git log` 没有任何提交，说明仓库还未 `git commit`，需要先执行 `git add` 与 `git commit`。若 `git branch` 中没有你准备推送的分支，请使用 `git checkout -b <新分支名>` 创建并提交。

### 5.2 执行推送

确认当前分支有需要提交的改动后，运行：

```bash
git push origin <分支名>
```

或在方案 B 中使用 `push-origin`：

```bash
git push push-origin <分支名>
```

其中的“分支名”指的是你当前希望推送到远端的 Git 分支，例如常见的 `main`、`master`，或你自行创建的功能分支如 `feature/refactor` 等。可通过以下命令确认分支名称：

```bash
git branch --show-current   # 显示当前所在分支
git branch                  # 列出本地全部分支
```

若看到报错 `error: src refspec <分支名> does not match any`，通常表示该分支在本地不存在或尚未提交任何内容。此时请重新检查分支名拼写，确保已在该分支上至少进行一次提交。

## 6. 清理令牌

推送完成后，建议执行：

```bash
unset GITHUB_TOKEN
```

必要时删除命令历史中的敏感条目，或在 GitHub 后台撤销该 PAT。

---

通过以上步骤，即可在本地终端正确配置 token 并解决 `not a git repository` 报错。
