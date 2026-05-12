# 将本项目上传到 GitHub（WSL 版）

本文档面向当前这个项目目录：`/mnt/f/pedcot`，并假设你正在 **WSL** 里操作。

目标是把本地项目一步一步上传到 GitHub。

## 先说结论

推荐做法：

- 在 WSL 里用 `git` 管理代码
- 用 **SSH** 连接 GitHub（后续最省事）
- 在 GitHub 上先创建一个**空仓库**
- 本地提交后，再 `git push` 到远程仓库

---

## 第 0 步：进入项目目录

先进入你的项目目录：

```bash
cd /mnt/f/pedcot
```

确认当前位置没错：

```bash
pwd
ls
```

你应该能看到类似这些文件：

- `README.md`
- `pyproject.toml`
- `src/`
- `tests/`
- `pedcot_overview.html`

---

## 第 1 步：确认 Git 已安装

运行：

```bash
git --version
```

如果能看到版本号，例如：

```text
git version 2.x.x
```

说明 Git 已经安装好。

如果提示没有安装，则执行：

```bash
sudo apt update
sudo apt install git -y
```

---

## 第 2 步：配置 Git 用户名和邮箱

如果这是你第一次在这台机器上使用 Git，需要先配置全局用户名和邮箱：

```bash
git config --global user.name "你的GitHub用户名或你的名字"
git config --global user.email "你的GitHub邮箱"
```

例如：

```bash
git config --global user.name "yourname"
git config --global user.email "you@example.com"
```

检查是否配置成功：

```bash
git config --global --list
```

---

## 第 3 步：初始化本地 Git 仓库

先确认当前目录还不是 Git 仓库时，再执行初始化：

```bash
git init
```

把默认分支设置为 `main`：

```bash
git branch -M main
```

检查状态：

```bash
git status
```

这时你会看到一批 `untracked files`，这是正常的。

---

## 第 4 步：确认哪些文件会被上传

这个项目已经有 `.gitignore` 文件，当前会忽略这些常见内容：

- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`
- `*.egg-info/`
- `artifacts/`
- `data/`

这意味着：

- 代码会上传
- `pedcot_overview.html` 会上传
- 虚拟环境不会上传
- 数据集目录 `data/` 不会上传
- 结果目录 `artifacts/` 不会上传

你可以用下面的命令确认即将提交的内容：

```bash
git status
```

如果你发现某些文件不想上传，就先修改 `.gitignore`，再继续下面的步骤。

---

## 第 5 步：把项目文件加入暂存区

把当前项目中的文件加入 Git 暂存区：

```bash
git add .
```

再次检查：

```bash
git status
```

如果显示很多绿色文件，说明这些文件已经准备提交。

---

## 第 6 步：创建第一次本地提交

执行：

```bash
git commit -m "Initial commit"
```

如果提交成功，说明你的本地 Git 仓库已经建立完成。

---

## 第 7 步：在 GitHub 上创建远程仓库

打开 GitHub 网站，然后按下面步骤操作：

1. 登录 GitHub
2. 点击右上角 `+`
3. 选择 `New repository`
4. 输入仓库名，例如：`pedcot`
5. 可选填写描述
6. 选择：
   - `Public`：公开仓库
   - `Private`：私有仓库
7. **不要勾选**：
   - `Add a README file`
   - `Add .gitignore`
   - `Choose a license`
8. 点击 `Create repository`

为什么要创建空仓库？

因为如果 GitHub 先自动帮你生成了 `README` 或别的文件，第一次推送时更容易出现冲突。

仓库创建完成后，GitHub 会给你一个仓库地址，通常有两种：

SSH 形式：

```text
git@github.com:你的用户名/pedcot.git
```

HTTPS 形式：

```text
https://github.com/你的用户名/pedcot.git
```

推荐使用 **SSH**。

---

## 第 8 步：检查本机是否已有 SSH Key

先检查是否已经有 SSH key：

```bash
ls ~/.ssh
```

如果你已经看到类似这些文件：

- `id_ed25519`
- `id_ed25519.pub`

说明你已经有 SSH key，可以直接跳到“第 10 步”。

如果没有，再继续下一步生成。

---

## 第 9 步：生成新的 SSH Key（如果还没有）

执行：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

例如：

```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

执行后会看到类似提示，通常一路回车即可：

- 保存位置直接回车，使用默认路径
- 是否设置 passphrase 可自行决定

默认会生成两个文件：

- 私钥：`~/.ssh/id_ed25519`
- 公钥：`~/.ssh/id_ed25519.pub`

---

## 第 10 步：启动 ssh-agent 并加载私钥

在 WSL 中执行：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

如果没有报错，说明私钥已经加载到 agent。

---

## 第 11 步：把 SSH 公钥添加到 GitHub

先在终端里输出公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出的整行内容。

如果你想在 **WSL** 里直接复制到 Windows 剪贴板，也可以用：

```bash
cat ~/.ssh/id_ed25519.pub | clip.exe
```

然后到 GitHub 网站执行：

1. 点击右上角头像
2. 打开 `Settings`
3. 在左侧找到 `SSH and GPG keys`
4. 点击 `New SSH key`
5. `Title` 随便写，例如：`WSL`
6. `Key type` 选择 `Authentication Key`
7. 把刚才复制的公钥粘贴进去
8. 点击 `Add SSH key`

---

## 第 12 步：测试 SSH 是否配置成功

运行：

```bash
ssh -T git@github.com
```

第一次连接时可能会提示是否信任 GitHub 主机，输入：

```text
yes
```

如果成功，通常会看到类似输出：

```text
Hi yourname! You've successfully authenticated, but GitHub does not provide shell access.
```

这表示 SSH 已经配置成功。

---

## 第 13 步：把本地仓库关联到 GitHub 远程仓库

假设你的 GitHub 用户名是 `yourname`，仓库名是 `pedcot`，运行：

```bash
git remote add origin git@github.com:yourname/pedcot.git
```

检查是否添加成功：

```bash
git remote -v
```

如果成功，你会看到类似：

```text
origin  git@github.com:yourname/pedcot.git (fetch)
origin  git@github.com:yourname/pedcot.git (push)
```

---

## 第 14 步：第一次推送到 GitHub

执行：

```bash
git push -u origin main
```

含义如下：

- `push`：把本地提交上传到 GitHub
- `-u`：建立本地 `main` 和远程 `origin/main` 的跟踪关系

第一次成功后，以后再推送通常只需要：

```bash
git push
```

推送完成后，刷新 GitHub 页面，就能看到你的项目已经上传成功。

---

## 第 15 步：以后更新项目时怎么继续上传

以后每次修改代码后，按下面顺序执行即可：

```bash
cd /mnt/f/pedcot
git status
git add .
git commit -m "写一句说明本次修改"
git push
```

例如：

```bash
git commit -m "Add project overview HTML page"
git push
```

---

## 如果你不想用 SSH，也可以用 HTTPS

如果你坚持使用 HTTPS，那么关联远程仓库时改成：

```bash
git remote add origin https://github.com/yourname/pedcot.git
```

然后推送：

```bash
git push -u origin main
```

注意：

- GitHub 已不支持用账号密码直接进行 Git 推送
- 你需要使用 **Personal Access Token (PAT)** 代替密码

如果你只是想最省事地长期使用，依然推荐 SSH。

---

## 常见问题排查

### 1）报错：`fatal: not a git repository`

原因：

- 你不在项目目录里
- 或者还没有执行 `git init`

解决：

```bash
cd /mnt/f/pedcot
git init
```

### 2）报错：`remote origin already exists`

说明你之前已经添加过远程地址。

查看当前远程地址：

```bash
git remote -v
```

如果要改成新的地址：

```bash
git remote set-url origin git@github.com:yourname/pedcot.git
```

### 3）报错：`Permission denied (publickey)`

说明 SSH 认证失败。请检查：

- SSH key 是否已经生成
- 公钥是否已经添加到 GitHub
- 是否执行过 `ssh-add ~/.ssh/id_ed25519`
- `ssh -T git@github.com` 是否能成功

### 4）报错：`failed to push some refs`

常见原因：

- GitHub 远程仓库里已经有内容
- 本地和远程分支历史不一致

最简单的避免方法：

- 创建 GitHub 仓库时不要自动初始化 `README`、`.gitignore` 或 `LICENSE`

### 5）为什么 `data/` 或 `artifacts/` 没上传？

因为当前项目的 `.gitignore` 已经忽略了这些目录：

- `data/`
- `artifacts/`

如果你真的想上传它们，需要先修改 `.gitignore`，然后重新 `git add`。

不过通常不建议把大数据或推理产物直接放进普通 Git 仓库。

---

## 最短命令版（适合已经理解流程后快速执行）

下面是一份最短版命令清单：

```bash
cd /mnt/f/pedcot
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
git init
git branch -M main
git add .
git commit -m "Initial commit"
```

然后在 GitHub 网站创建空仓库后，继续执行：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
git remote add origin git@github.com:你的用户名/pedcot.git
git push -u origin main
```

---

## 官方文档参考

以下步骤参考了 GitHub 官方文档，并结合你当前的 WSL 使用场景进行了整理：

- 添加本地项目到 GitHub：  
  <https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows>
- 生成 SSH key 并加入 ssh-agent：  
  <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>
- 添加 SSH key 到 GitHub 账户：  
  <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?tool=webui>

