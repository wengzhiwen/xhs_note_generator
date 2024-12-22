# 小红书笔记生成器 (XHS Note Generator)

这是一个fork项目，追加本地视频的处理功能。

 - 顺便，把一个UNSPLASH回调地址的强制验证给去掉了，似乎用不到
 - 顺便，我把温度和MAX TOKEN的配置不生效的bug给修了
 - 顺便，我把生成用的模型放到配置文件里了，我测下来qwen/qwq-32b-preview价格便宜效果也不错。刚刚推出的gemini2现在（当前）一定使用量下免费，用来调整还是很合适的

## 👤 原项目

- **GitHub**：[whotto/Video_note_generator](https://github.com/whotto/Video_note_generator)

## 干货

0. 环境配置等准备工作见原项目，OpenRouter记得充点钱

1. 准备要处理的视频，经过实践：视频里面旁白内容要丰富一点，全是BGM的视频没啥用

2. 根据sample.json改一个你自己的json文件

## 使用

运行生成器：
```bash
python video_note_generator.py sample.json
```

