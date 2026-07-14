# GitHub 常用指令说明

这个文件用来说明日常把本地代码上传到 GitHub 时最常用的一组 Git 指令。

## 1. 查看当前仓库状态

```bash
git status
```

作用：查看哪些文件被修改、新增、删除，以及当前所在分支。

常用简洁版：

```bash
git status -sb
```

输出里常见标记：

- `M`：文件被修改。
- `D`：文件被删除。
- `??`：新文件，还没有被 Git 跟踪。

## 2. 查看远程仓库地址

```bash
git remote -v
```

作用：查看当前本地仓库连接到了哪些远程仓库。

常见输出：

```text
origin  https://github.com/user/repo.git (fetch)
origin  https://github.com/user/repo.git (push)
```

`origin` 是远程仓库的默认名字，后面的地址就是 GitHub 仓库地址。

## 3. 添加远程 GitHub 仓库

```bash
git remote add origin https://github.com/user/repo.git
```

作用：把本地仓库和 GitHub 仓库关联起来。

如果已经有 `origin`，想改成新的 GitHub 地址：

```bash
git remote set-url origin https://github.com/user/repo.git
```

## 4. 添加文件到暂存区

添加单个文件：

```bash
git add test/github_common_commands.md
```

添加整个文件夹：

```bash
git add test/
```

添加当前目录下所有变化：

```bash
git add -A
```

注意：`git add -A` 会把所有修改、新增、删除都加入提交。工作区有很多无关改动时，不建议直接用它。

## 5. 提交代码

```bash
git commit -m "add github commands guide"
```

作用：把暂存区里的改动保存成一次本地提交。

提交信息应该简短说明这次改了什么。

## 6. 推送到 GitHub

```bash
git push origin master
```

或者当前分支叫 `main` 时：

```bash
git push origin main
```

第一次推送新分支时常用：

```bash
git push -u origin branch-name
```

`-u` 的作用是把本地分支和远程分支绑定起来。以后可以直接执行：

```bash
git push
```

## 7. 拉取 GitHub 上的最新代码

```bash
git pull
```

作用：把远程仓库的最新提交拉到本地，并尝试自动合并。

更完整的写法：

```bash
git pull origin main
```

## 8. 创建和切换分支

创建并切换到新分支：

```bash
git checkout -b feature/test-folder
```

较新的 Git 也可以用：

```bash
git switch -c feature/test-folder
```

切换已有分支：

```bash
git switch main
```

查看所有本地分支：

```bash
git branch
```

## 9. 查看提交历史

```bash
git log
```

简洁版：

```bash
git log --oneline
```

查看最近一次提交：

```bash
git log -1 --oneline
```

## 10. 查看文件具体改动

查看还没有暂存的改动：

```bash
git diff
```

查看已经 `git add` 但还没有提交的改动：

```bash
git diff --cached
```

## 11. 撤销常见操作

撤销某个文件未暂存的修改：

```bash
git restore file.py
```

把已经 `git add` 的文件从暂存区拿出来，但保留文件内容：

```bash
git restore --staged file.py
```

删除最近一次提交，但保留代码改动：

```bash
git reset --soft HEAD~1
```

注意：不要随便使用 `git reset --hard`，它会丢弃本地改动。

## 12. 推荐的日常上传流程

```bash
git status -sb
git add test/github_common_commands.md
git commit -m "add github commands guide"
git push origin main
```

如果你的默认分支是 `master`，最后一行改成：

```bash
git push origin master
```

## 13. 这次新增 test 文件夹的流程

这次我们新增的是：

```text
test/github_common_commands.md
```

本地 Git 提交流程可以是：

```bash
git status -sb
git add test/github_common_commands.md
git commit -m "add github commands guide"
git push origin main
```

核心思想是：

1. `git status` 先确认改了什么。
2. `git add` 只选择这次要提交的文件。
3. `git commit` 在本地形成一次提交。
4. `git push` 把提交上传到 GitHub。
