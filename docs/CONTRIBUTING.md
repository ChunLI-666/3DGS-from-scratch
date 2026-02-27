# Contributing to 3DGS-from-scratch

感谢你对本项目的关注！欢迎任何形式的贡献。

## How to Contribute

### Fork & Pull Request

1. Fork 本仓库到你的 GitHub 账号
2. Clone 你的 fork：
   ```bash
   git clone https://github.com/<your-username>/3DGS-from-scratch.git
   cd 3DGS-from-scratch
   ```
3. 创建新分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. 完成修改后提交：
   ```bash
   git add <changed-files>
   git commit -m "feat(navi): [PRO-1173] Your concise description"
   ```
5. Push 并创建 Pull Request：
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在 GitHub 上创建 PR，描述你的修改内容。

## Code Style

- **遵循现有代码风格**：查看同目录下已有文件的写法，保持一致。
- **注释语言**：中文注释 + English technical terms（中英混合）。
- **命名规范**：使用清晰可读的变量名，避免单字母变量（循环变量除外）。
- **缩进**：Python 使用 4 空格缩进。
- **函数设计**：保持函数单一职责，嵌套层级不超过 3 层。

## Notebook Style Conventions

所有 Jupyter Notebook 需遵循以下规范：

1. **Colab Badge**：每个 notebook 顶部添加 Open In Colab 徽章
2. **环境配置 Cell**：第一个代码 cell 负责安装依赖和环境配置
3. **总结 Cell**：最后一个 cell 为本 notebook 的学习总结
4. **中英混合注释**：Markdown cell 使用中文解释 + 英文术语
5. **可视化**：使用 matplotlib 进行图表展示
6. **自包含**：每个 cell 的 import 应自包含，不依赖其他 cell 的隐式导入

## Running Tests

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试（如果有）
pytest tests/ -v

# 验证 notebook 可执行
jupyter nbconvert --execute --to notebook notebooks/phase1/00_environment_setup.ipynb
```

## Reporting Issues

- 使用 GitHub Issues 报告 bug 或提出功能请求
- 请附上复现步骤、错误信息和运行环境信息

## License

本项目采用 MIT License，详见 [LICENSE](../LICENSE)。
