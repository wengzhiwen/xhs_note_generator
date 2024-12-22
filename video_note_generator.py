import os
import sys
import json
import time
import shutil
import re
import subprocess
from typing import Dict, List, Optional, Tuple
import datetime
from pathlib import Path
import random
from itertools import zip_longest

import yt_dlp
import httpx
from unsplash.api import Api as UnsplashApi
from unsplash.auth import Auth as UnsplashAuth
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import whisper
import openai
import argparse

# 加载环境变量
load_dotenv()

# 检查必要的环境变量
required_env_vars = {
    'OPENROUTER_API_KEY': '用于OpenRouter API',
    'OPENROUTER_API_URL': '用于OpenRouter API',
    'OPENROUTER_APP_NAME': '用于OpenRouter API',
    'OPENROUTER_HTTP_REFERER': '用于OpenRouter API',
    'OPENROUTER_MODLE': '用于OpenRouter API',
    'UNSPLASH_ACCESS_KEY': '用于图片搜索',
    'UNSPLASH_SECRET_KEY': '用于Unsplash认证'
}

missing_env_vars = []
for var, desc in required_env_vars.items():
    if not os.getenv(var):
        missing_env_vars.append(f"  - {var} ({desc})")

if missing_env_vars:
    print("注意：以下环境变量未设置：")
    print("\n".join(missing_env_vars))
    print("\n将使用基本功能继续运行（无AI优化和图片）。")
    print("如需完整功能，请在 .env 文件中设置相应的 API 密钥。")
    print("继续处理...\n")

# 配置代理
http_proxy = os.getenv('HTTP_PROXY')
https_proxy = os.getenv('HTTPS_PROXY')
proxies = {
    'http': http_proxy,
    'https': https_proxy
} if http_proxy and https_proxy else None

# 禁用 SSL 验证（仅用于开发环境）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# OpenRouter configuration
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
openrouter_app_name = os.getenv('OPENROUTER_APP_NAME', 'video_note_generator')
openrouter_http_referer = os.getenv('OPENROUTER_HTTP_REFERER', 'https://github.com')
openrouter_available = False
openrouter_max_tokens = os.getenv('MAX_TOKENS', 4000)
try:
    openrouter_temperature = float(os.getenv('TEMPERATURE', 0.7))
except ValueError:
    print("⚠️ TEMPERATURE 环境变量配置不正确，使用默认值 0.7")
    openrouter_temperature = 0.7

# 配置 OpenAI API
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": openrouter_http_referer,
        "X-Title": openrouter_app_name,
    }
)

# 选择要使用的模型
AI_MODEL = os.getenv("OPENROUTER_MODLE", "google/gemini-pro")

# Test OpenRouter connection
if openrouter_api_key:
    try:
        print(f"正在测试 OpenRouter API {AI_MODEL} 连接...")
        
        # 测试API连接
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "这是一个链接测试的一部分，你只需要正常表现让我们知道你很好就可以了"},
                {"role": "user", "content": "请问你是谁"}
            ],
            temperature=openrouter_temperature,
            max_tokens=100 # 只是测试链接性，限制最大输出长度
        )
        if not response.choices:
            print(response)
            raise Exception("API 连接测试失败")

        print(f"✅ OpenRouter API 连接测试成功")
        openrouter_available = True
    except Exception as e:
        print(f"⚠️ OpenRouter API 连接测试失败: {str(e)}")
        print("将继续尝试使用API，但可能会遇到问题")

# 检查Unsplash配置
unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY')
unsplash_client = None

if unsplash_access_key:
    try:
        auth = UnsplashAuth(
            client_id=unsplash_access_key,
            client_secret=None,
            redirect_uri=None
        )
        unsplash_client = UnsplashApi(auth)
        print("✅ Unsplash API 配置成功")
    except Exception as e:
        print(f"❌ Failed to initialize Unsplash client: {str(e)}")

# 检查ffmpeg
ffmpeg_path = None
try:
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)
    print("✅ ffmpeg is available at /opt/homebrew/bin/ffmpeg")
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
except Exception:
    try:
        subprocess.run(["ffmpeg", "-version"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        print("✅ ffmpeg is available (from PATH)")
        ffmpeg_path = "ffmpeg"
    except Exception as e:
        print(f"⚠️ ffmpeg not found: {str(e)}")

class DownloadError(Exception):
    """自定义下载错误类"""
    def __init__(self, message: str, platform: str, error_type: str, details: str = None):
        self.message = message
        self.platform = platform
        self.error_type = error_type
        self.details = details
        super().__init__(self.message)

class VideoNoteGenerator:
    def __init__(self, output_dir: str = "temp_notes"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.openrouter_available = openrouter_available
        self.unsplash_client = unsplash_client
        self.ffmpeg_path = ffmpeg_path
        
        # 初始化whisper模型
        print("正在加载Whisper模型...")
        self.whisper_model = None
        try:
            self.whisper_model = whisper.load_model("medium")
            print("✅ Whisper模型加载成功")
        except Exception as e:
            print(f"⚠️ Whisper模型加载失败: {str(e)}")
            print("将在需要时重试加载")
        
        # 日志目录
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # cookie目录
        self.cookie_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cookies')
        os.makedirs(self.cookie_dir, exist_ok=True)
        
        # 平台cookie文件
        self.platform_cookies = {
            'douyin': os.path.join(self.cookie_dir, 'douyin_cookies.txt'),
            'bilibili': os.path.join(self.cookie_dir, 'bilibili_cookies.txt'),
            'youtube': os.path.join(self.cookie_dir, 'youtube_cookies.txt')
        }
    
    def _ensure_whisper_model(self) -> None:
        """确保Whisper模型已加载"""
        if self.whisper_model is None:
            try:
                print("正在加载Whisper模型...")
                self.whisper_model = whisper.load_model("medium")
                print("✅ Whisper模型加载成功")
            except Exception as e:
                print(f"⚠️ Whisper模型加载失败: {str(e)}")

    def _determine_platform(self, url: str) -> Optional[str]:
        """
        确定视频平台
        
        Args:
            url: 视频URL
            
        Returns:
            str: 平台名称 ('youtube', 'douyin', 'bilibili') 或 None
        """
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'douyin.com' in url:
            return 'douyin'
        elif 'bilibili.com' in url:
            return 'bilibili'
        return None

    def _handle_download_error(self, error: Exception, platform: str, url: str) -> str:
        """
        处理下载错误并返回用户友好的错误消息
        
        Args:
            error: 异常对象
            platform: 平台名称
            url: 视频URL
            
        Returns:
            str: 用户友好的错误消息
        """
        error_msg = str(error)
        
        if "SSL" in error_msg:
            return "⚠️ SSL证书验证失败，请检查网络连接"
        elif "cookies" in error_msg.lower():
            return f"⚠️ {platform}访问被拒绝，可能需要更新cookie或更换IP地址"
        elif "404" in error_msg:
            return "⚠️ 视频不存在或已被删除"
        elif "403" in error_msg:
            return "⚠️ 访问被拒绝，可能需要登录或更换IP地址"
        elif "unavailable" in error_msg.lower():
            return "⚠️ 视频当前不可用，可能是地区限制或版权问题"
        else:
            return f"⚠️ 下载失败: {error_msg}"

    def _get_platform_options(self, platform: str) -> Dict:
        """获取平台特定的下载选项"""
        # 基本选项
        options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': '%(title)s.%(ext)s'
        }
        
        if platform in self.platform_cookies and os.path.exists(self.platform_cookies[platform]):
            options['cookiefile'] = self.platform_cookies[platform]
            
        return options

    def _validate_cookies(self, platform: str) -> bool:
        """验证cookie是否有效"""
        if platform not in self.platform_cookies:
            return False
        
        cookie_file = self.platform_cookies[platform]
        return os.path.exists(cookie_file)

    def _get_alternative_download_method(self, platform: str, url: str) -> Optional[str]:
        """获取备用下载方法"""
        if platform == 'youtube':
            return 'pytube'
        elif platform == 'douyin':
            return 'requests'
        elif platform == 'bilibili':
            return 'you-get'
        return None

    def _download_with_alternative_method(self, platform: str, url: str, temp_dir: str, method: str) -> Optional[str]:
        """使用备用方法下载"""
        try:
            if method == 'you-get':
                cmd = ['you-get', '--no-proxy', '--no-check-certificate', '-o', temp_dir, url]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # 查找下载的文件
                    files = [f for f in os.listdir(temp_dir) if f.endswith(('.mp4', '.flv', '.webm'))]
                    if files:
                        return os.path.join(temp_dir, files[0])
                raise Exception(result.stderr)
                
            elif method == 'requests':
                # 使用requests直接下载
                headers = {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                # 首先获取页面内容
                response = httpx.get(url, headers=headers, verify=False)
                
                if response.status_code == 200:
                    # 尝试从页面中提取视频URL
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    video_url = None
                    # 查找video标签
                    video_tags = soup.find_all('video')
                    for video in video_tags:
                        src = video.get('src') or video.get('data-src')
                        if src:
                            video_url = src
                            break
                    
                    if not video_url:
                        # 尝试查找其他可能包含视频URL的元素
                        import re
                        video_patterns = [
                            r'https?://[^"\'\s]+\.(?:mp4|m3u8)[^"\'\s]*',
                            r'playAddr":"([^"]+)"',
                            r'play_url":"([^"]+)"'
                        ]
                        for pattern in video_patterns:
                            matches = re.findall(pattern, response.text)
                            if matches:
                                video_url = matches[0]
                                break
                    
                    if video_url:
                        if not video_url.startswith('http'):
                            video_url = 'https:' + video_url if video_url.startswith('//') else video_url
                        
                        # 下载视频
                        video_response = httpx.get(video_url, headers=headers, stream=True, verify=False)
                        if video_response.status_code == 200:
                            file_path = os.path.join(temp_dir, 'video.mp4')
                            with open(file_path, 'wb') as f:
                                for chunk in video_response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            return file_path
                        
                    raise Exception(f"无法下载视频: HTTP {video_response.status_code}")
                raise Exception(f"无法访问页面: HTTP {response.status_code}")
                
            elif method == 'pytube':
                # 禁用SSL验证
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                
                from pytube import YouTube
                yt = YouTube(url)
                # 获取最高质量的MP4格式视频
                video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if video:
                    return video.download(output_path=temp_dir)
                raise Exception("未找到合适的视频流")
                
        except Exception as e:
            print(f"备用下载方法 {method} 失败: {str(e)}")
            return None

    def _download_video(self, url: str, temp_dir: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
        """下载视频并返回音频文件路径和信息"""
        try:
            platform = self._determine_platform(url)
            if not platform:
                raise DownloadError("不支持的视频平台", "unknown", "platform_error")

            # 基本下载选项
            options = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'quiet': True,
                'no_warnings': True,
            }

            # 下载视频
            for attempt in range(3):  # 最多重试3次
                try:
                    with yt_dlp.YoutubeDL(options) as ydl:
                        print(f"正在尝试下载（第{attempt + 1}次）...")
                        info = ydl.extract_info(url, download=True)
                        if not info:
                            raise DownloadError("无法获取视频信息", platform, "info_error")

                        # 找到下载的音频文件
                        downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
                        if not downloaded_files:
                            raise DownloadError("未找到下载的音频文件", platform, "file_error")

                        audio_path = os.path.join(temp_dir, downloaded_files[0])
                        if not os.path.exists(audio_path):
                            raise DownloadError("音频文件不存在", platform, "file_error")

                        video_info = {
                            'title': info.get('title', '未知标题'),
                            'uploader': info.get('uploader', '未知作者'),
                            'description': info.get('description', ''),
                            'duration': info.get('duration', 0),
                            'platform': platform
                        }

                        print(f"✅ {platform}视频下载成功")
                        return audio_path, video_info

                except Exception as e:
                    print(f"⚠️ 下载失败（第{attempt + 1}次）: {str(e)}")
                    if attempt < 2:  # 如果不是最后一次尝试
                        print("等待5秒后重试...")
                        time.sleep(5)
                    else:
                        raise  # 最后一次失败，抛出异常

        except Exception as e:
            error_msg = self._handle_download_error(e, platform, url)
            print(f"⚠️ {error_msg}")
            return None, None

    def _transcribe_audio(self, audio_path: str) -> str:
        """使用Whisper转录音频"""
        try:
            self._ensure_whisper_model()
            if not self.whisper_model:
                raise Exception("Whisper模型未加载")
                
            print("正在转录音频（这可能需要几分钟）...")
            result = self.whisper_model.transcribe(
                audio_path,
                language='zh',  # 指定中文
                task='transcribe',
                best_of=5,
                initial_prompt="以下是一段视频的转录内容。请用流畅的中文输出。"  # 添加中文提示
            )
            return result["text"].strip()
            
        except Exception as e:
            print(f"⚠️ 音频转录失败: {str(e)}")
            return ""

    def _organize_content(self, content: str) -> str:
        """使用AI整理内容"""
        try:
            if not self.openrouter_available:
                print("⚠️ OpenRouter API 未配置，将返回原始内容")
                return content

            # 构建系统提示词
            system_prompt = """你是一位著名的科普作家和博客作者，著作等身，屡获殊荣，尤其在内容创作领域有深厚的造诣。

请使用 4C 模型（建立联系 Connection、展示冲突 Conflict、强调改变 Change、即时收获 Catch）为转录的文字内容创建结构。

写作要求：
- 从用户的问题出发，引导读者理解核心概念及其背景
- 使用第二人称与读者对话，语气亲切平实
- 确保所有观点和内容基于用户提供的转录文本
- 如无具体实例，则不编造
- 涉及复杂逻辑时，使用直观类比
- 避免内容重复冗余
- 逻辑递进清晰，从问题开始，逐步深入

Markdown格式要求：
- 大标题突出主题，吸引眼球，最好使用疑问句
- 小标题简洁有力，结构清晰，尽量使用单词或短语
- 直入主题，在第一部分清晰阐述问题和需求
- 正文使用自然段，避免使用列表形式
- 内容翔实，避免过度简略，特别注意保留原文中的数据和示例信息
- 如有来源URL，使用文内链接形式
- 保留原文中的Markdown格式图片链接"""

            # 构建用户提示词
            final_prompt = f"""请根据以下转录文字内容，创作一篇结构清晰、易于理解的博客文章。

转录文字内容：

{content}"""

            # 调用API
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=openrouter_temperature,
                max_tokens=openrouter_max_tokens
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            
            return content

        except Exception as e:
            print(f"⚠️ 内容整理失败: {str(e)}")
            return content

    def split_content(self, text: str, max_chars: int = 2000) -> List[str]:
        """按段落分割文本，保持上下文的连贯性
        
        特点：
        1. 保持段落完整性：不会在段落中间断开
        2. 保持句子完整性：确保句子不会被截断
        3. 添加重叠内容：每个chunk都包含上一个chunk的最后一段
        4. 智能分割：对于超长段落，按句子分割并保持完整性
        """
        if not text:
            return []

        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        last_paragraph = None  # 用于存储上一个chunk的最后一段
        
        for para in paragraphs:
            para = para.strip()
            if not para:  # 跳过空段落
                continue
            
            para_length = len(para)
            
            # 如果这是新chunk的开始，且有上一个chunk的最后一段，添加它作为上下文
            if not current_chunk and last_paragraph:
                current_chunk.append(f"上文概要：\n{last_paragraph}\n")
                current_length += len(last_paragraph) + 20  # 加上标题的长度
            
            # 如果单个段落就超过了最大长度，需要按句子分割
            if para_length > max_chars:
                # 如果当前块不为空，先保存
                if current_chunk:
                    last_paragraph = current_chunk[-1]
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    if last_paragraph:
                        current_chunk.append(f"上文概要：\n{last_paragraph}\n")
                        current_length += len(last_paragraph) + 20
                
                # 按句子分割长段落
                sentences = re.split(r'([。！？])', para)
                current_sentence = []
                current_sentence_length = 0
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    # 如果有标点符号，加上标点
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                    
                    # 如果加上这个句子会超过最大长度，保存当前块并开始新块
                    if current_sentence_length + len(sentence) > max_chars and current_sentence:
                        chunks.append(''.join(current_sentence))
                        current_sentence = [sentence]
                        current_sentence_length = len(sentence)
                    else:
                        current_sentence.append(sentence)
                        current_sentence_length += len(sentence)
                
                # 保存最后一个句子块
                if current_sentence:
                    chunks.append(''.join(current_sentence))
            else:
                # 如果加上这个段落会超过最大长度，保存当前块并开始新块
                if current_length + para_length > max_chars and current_chunk:
                    last_paragraph = current_chunk[-1]
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    if last_paragraph:
                        current_chunk.append(f"上文概要：\n{last_paragraph}\n")
                        current_length += len(last_paragraph) + 20
                current_chunk.append(para)
                current_length += para_length
        
        # 保存最后一个块
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def _organize_long_content(self, content: str, duration: int = 0) -> str:
        """使用AI整理长文内容"""
        if not content.strip():
            return ""
        
        if not self.openrouter_available:
            print("⚠️ OpenRouter API 不可用，将返回原始内容")
            return content
        
        content_chunks = self.split_content(content)
        organized_chunks = []
        
        print(f"内容将分为 {len(content_chunks)} 个部分进行处理...")
        
        for i, chunk in enumerate(content_chunks, 1):
            print(f"正在处理第 {i}/{len(content_chunks)} 部分...")
            organized_chunk = self._organize_content(chunk)
            organized_chunks.append(organized_chunk)
    
        return "\n\n".join(organized_chunks)

    def convert_to_xiaohongshu(self, content: str) -> Tuple[str, List[str], List[str], List[str]]:
        """将博客文章转换为小红书风格的笔记，并生成标题和标签"""
        try:
            if not self.openrouter_available:
                print("⚠️ OpenRouter API 未配置，将返回原始内容")
                return content, [], [], []

            # 构建系统提示词
            system_prompt = """你是一位专业的小红书爆款文案写作大师，擅长将普通内容转换为刷屏级爆款笔记。
请将输入的内容转换为小红书风格的笔记，需要满足以下要求：

1. 标题创作（重要‼️）：
- 二极管标题法：
  * 追求快乐：产品/方法 + 只需N秒 + 逆天效果
  * 逃避痛苦：不采取行动 + 巨大损失 + 紧迫感
- 爆款关键词（必选1-2个）：
  * 高转化词：好用到哭、宝藏、神器、压箱底、隐藏干货、高级感
  * 情感词：绝绝子、破防了、治愈、万万没想到、爆款、永远可以相信
  * 身份词：小白必看、手残党必备、打工人、普通女生
  * 程度词：疯狂点赞、超有料、无敌、一百分、良心推荐
- 标题规则：
  * 字数：20字以内
  * emoji：2-4个相关表情
  * 标点：感叹号、省略号增强表达
  * 风格：口语化、制造悬念

2. 正文创作：
- 开篇设置（抓住痛点）：
  * 共情开场：描述读者痛点
  * 悬念引导：埋下解决方案的伏笔
  * 场景还原：具体描述场景
- 内容结构：
  * 每段开头用emoji引导
  * 重点内容加粗突出
  * 适当空行增加可读性
  * 步骤说明要清晰
- 写作风格：
  * 热情亲切的语气
  * 大量使用口语化表达
  * 插入互动性问句
  * 加入个人经验分享
- 高级技巧：
  * 使用平台热梗
  * 加入流行口头禅
  * 设置悬念和爆点
  * 情感共鸣描写

3. 标签优化：
- 提取4类标签（每类1-2个）：
  * 核心关键词：主题相关
  * 关联关键词：长尾词
  * 高转化词：购买意向强
  * 热搜词：行业热点

4. 整体要求：
- 内容体量：根据内容自动调整
- 结构清晰：善用分点和空行
- 情感真实：避免过度营销
- 互动引导：设置互动机会
- AI友好：避免机器味

注意：创作时要始终记住，标题决定打开率，内容决定完播率，互动决定涨粉率！"""

            # 构建用户提示词
            user_prompt = f"""请将以下内容转换为爆款小红书笔记。

内容如下：
{content}

请按照以下格式返回：
1. 第一行：爆款标题（遵循二极管标题法，必须有emoji）
2. 空一行
3. 正文内容（注意结构、风格、技巧的运用，控制在600-800字之间）
4. 空一行
5. 标签列表（每类标签都要有，用#号开头）

创作要求：
1. 标题要让人忍不住点进来看
2. 内容要有干货，但表达要轻松
3. 每段都要用emoji装饰
4. 标签要覆盖核心词、关联词、转化词、热搜词
5. 设置2-3处互动引导
6. 通篇要有感情和温度
7. 正文控制在600-800字之间

"""

            # 调用API
            print(f"开始调用{AI_MODEL}...")
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=openrouter_temperature,
                max_tokens=openrouter_max_tokens
            )
            
            if not response.choices:
                raise Exception("API 返回结果为空")

            # 处理返回的内容
            xiaohongshu_content = response.choices[0].message.content.strip()
            print(f"\n📝 API返回内容：\n{xiaohongshu_content}\n")
            
            # 提取标题（第一行）
            content_lines = xiaohongshu_content.split('\n')
            titles = []
            for line in content_lines:
                line = line.strip()
                if line and not line.startswith('#') and '：' not in line and '。' not in line:
                    titles = [line]
                    break
            
            if not titles:
                print("⚠️ 未找到标题，尝试其他方式提取...")
                # 尝试其他方式提取标题
                title_match = re.search(r'^[^#\n]+', xiaohongshu_content)
                if title_match:
                    titles = [title_match.group(0).strip()]
            
            if titles:
                print(f"✅ 提取到标题: {titles[0]}")
            else:
                print("⚠️ 未能提取到标题")
            
            # 提取标签（查找所有#开头的标签）
            tags = []
            tag_matches = re.findall(r'#([^\s#]+)', xiaohongshu_content)
            if tag_matches:
                tags = tag_matches
                print(f"✅ 提取到{len(tags)}个标签")
            else:
                print("⚠️ 未找到标签")
            
            # 获取相关图片
            images = []
            if self.unsplash_client:
                # 使用标题和标签作为搜索关键词
                search_terms = titles + tags[:2] if tags else titles
                search_query = ' '.join(search_terms)
                try:
                    images = self._get_unsplash_images(search_query, count=4)
                    if images:
                        print(f"✅ 成功获取{len(images)}张配图")
                    else:
                        print("⚠️ 未找到相关配图")
                except Exception as e:
                    print(f"⚠️ 获取配图失败: {str(e)}")
            
            return xiaohongshu_content, titles, tags, images

        except Exception as e:
            print(f"⚠️ 转换小红书笔记失败: {str(e)}")
            return content, [], [], []

    def _get_unsplash_images(self, query: str, count: int = 3) -> List[str]:
        """从Unsplash获取相关图片"""
        if not self.unsplash_client:
            print("⚠️ Unsplash客户端未初始化")
            return []
            
        try:
            # 将查询词翻译成英文以获得更好的结果
            if self.openrouter_available:
                try:
                    response = client.chat.completions.create(
                        model=AI_MODEL,
                        messages=[
                            {"role": "system", "content": "你是一个翻译助手。请将输入的中文关键词翻译成最相关的1-3个英文关键词，用逗号分隔。直接返回翻译结果，不要加任何解释。例如：\n输入：'保险理财知识'\n输出：insurance,finance,investment"},
                            {"role": "user", "content": query}
                        ],
                        temperature=0.3, # 翻译场景下使用特殊的设定
                        max_tokens=50 # 翻译场景下使用特殊的设定
                    )
                    if response.choices:
                        query = response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"⚠️ 翻译关键词失败: {str(e)}")
            
            # 使用httpx直接调用Unsplash API
            headers = {
                'Authorization': f'Client-ID {os.getenv("UNSPLASH_ACCESS_KEY")}'
            }
            
            # 对每个关键词分别搜索
            all_photos = []
            for keyword in query.split(','):
                response = httpx.get(
                    'https://api.unsplash.com/search/photos',
                    params={
                        'query': keyword.strip(),
                        'per_page': count,
                        'orientation': 'portrait',  # 小红书偏好竖版图片
                        'content_filter': 'high'    # 只返回高质量图片
                    },
                    headers=headers,
                    verify=False  # 禁用SSL验证
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data['results']:
                        # 获取图片URL，优先使用regular尺寸
                        photos = [photo['urls'].get('regular', photo['urls']['small']) 
                                for photo in data['results']]
                        all_photos.extend(photos)
            
            # 如果收集到的图片不够，用最后一个关键词继续搜索
            while len(all_photos) < count and query:
                response = httpx.get(
                    'https://api.unsplash.com/search/photos',
                    params={
                        'query': query.split(',')[-1].strip(),
                        'per_page': count - len(all_photos),
                        'orientation': 'portrait',
                        'content_filter': 'high',
                        'page': 2  # 获取下一页的结果
                    },
                    headers=headers,
                    verify=False
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data['results']:
                        photos = [photo['urls'].get('regular', photo['urls']['small']) 
                                for photo in data['results']]
                        all_photos.extend(photos)
                    else:
                        break
                else:
                    break
            
            # 返回指定数量的图片
            return all_photos[:count]
            
        except Exception as e:
            print(f"⚠️ 获取图片失败: {str(e)}")
            return []

    def process_video(self, url: str, title:str = None, content:str = None) -> List[str]:
        """处理视频链接，生成笔记
        
        Args:
            url (str): 视频链接
        
        Returns:
            List[str]: 生成的笔记文件路径列表
        """
        print("\n📹 正在处理视频...")
        
        # 创建临时目录
        temp_dir = os.path.join(self.output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            if title is not None and content is not None:
                # 说明是本地视频文件不需要下载
                # url就是视频文件路径
                video_path = url
                if not os.path.exists(video_path):
                    raise Exception("视频文件不存在")
                # 将视频文件直接复制到临时目录
                shutil.copy(video_path, temp_dir)
                # 更新video_path为临时目录中的视频文件
                video_path = os.path.join(temp_dir, os.path.basename(video_path))
                
                # 准备在临时目录中以视频文件名为基础创建音频文件
                audio_path = os.path.join(temp_dir, os.path.basename(video_path) + '.mp3')
                
                # 使用ffmpeg_path中的ffmpeg提取音频，也放到临时目录
                try:
                    command = [
                        self.ffmpeg_path,
                        '-i', video_path,           # 输入文件
                        '-vn',                      # 禁用视频
                        '-acodec', 'libmp3lame',    # 音频编码器
                        '-q:a', '2',                # 音质设置 (0-9, 2=高质量)
                        audio_path                    # 输出文件
                    ]
                    subprocess.run(command, check=True)
                    print(f"Audio extracted to: {temp_dir}")
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting audio: {e}")
                     
                # 获取视频的长度
                duration = 0.0
                
                ffprobe_path = 'ffprobe'
                # 将ffmepg_path中的ffmpge路径替换为ffprobe
                if self.ffmpeg_path:
                    ffprobe_path = self.ffmpeg_path.replace('ffmpeg', 'ffprobe')
                
                try:
                    command = [
                        ffprobe_path,
                        '-v', 'quiet',
                        '-print_format', 'json',
                        '-show_format',
                        '-show_streams',
                        video_path
                    ]
                    output = subprocess.check_output(command)
                    data = json.loads(output)
                    duration = float(data['format']['duration'])
                except Exception as e:
                    print(f"Error getting duration, continue: {e}")
                    return 0.0
                
                video_info = {
                    'title': title,
                    'uploader': '未知作者',
                    'description': content,
                    'duration': duration,
                    'platform': 'local'
                }
            else:
                # 下载视频
                print("⬇️ 正在下载视频...")
                result = self._download_video(url, temp_dir)
                if not result:
                    return []
                    
                audio_path, video_info = result
                
            if not audio_path or not video_info:
                return []
                
            print(f"✅ 视频下载成功: {video_info['title']}")
            
            # 转录音频
            print("\n🎙️ 正在转录音频...")
            print("正在转录音频（这可能需要几分钟）...")
            transcript = self._transcribe_audio(audio_path)
            if not transcript:
                return []

            # 保存原始转录内容
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_file = os.path.join(self.output_dir, f"{timestamp}_original.md")
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(f"# {video_info['title']}\n\n")
                f.write(f"## 视频信息\n")
                f.write(f"- 作者：{video_info['uploader']}\n")
                f.write(f"- 时长：{video_info['duration']}秒\n")
                f.write(f"- 平台：{video_info['platform']}\n")
                f.write(f"- 链接：{url}\n\n")
                f.write(f"## 原始转录内容\n\n")
                f.write(transcript)

            # 整理长文版本
            print("\n📝 正在整理长文版本...")
            organized_content = self._organize_long_content(transcript, video_info['duration'])
            organized_file = os.path.join(self.output_dir, f"{timestamp}_organized.md")
            with open(organized_file, 'w', encoding='utf-8') as f:
                f.write(f"# {video_info['title']} - 整理版\n\n")
                f.write(f"## 视频信息\n")
                f.write(f"- 作者：{video_info['uploader']}\n")
                f.write(f"- 时长：{video_info['duration']}秒\n")
                f.write(f"- 平台：{video_info['platform']}\n")
                f.write(f"- 链接：{url}\n\n")
                f.write(f"## 内容整理\n\n")
                f.write(organized_content)
            
            # 生成小红书版本
            print("\n📱 正在生成小红书版本...")
            try:
                xiaohongshu_content, titles, tags, images = self.convert_to_xiaohongshu(organized_content)
                
                # 保存小红书版本
                xiaohongshu_file = os.path.join(self.output_dir, f"{timestamp}_xiaohongshu.md")
                
                # 写入文件
                with open(xiaohongshu_file, "w", encoding="utf-8") as f:
                    # 写入标题
                    f.write(f"# {titles[0]}\n\n")
                    
                    # 如果有图片，先写入第一张作为封面
                    if images:
                        f.write(f"![封面图]({images[0]})\n\n")
                    
                    # 写入正文内容的前半部分
                    content_parts = xiaohongshu_content.split('\n\n')
                    mid_point = len(content_parts) // 2
                    
                    # 写入前半部分
                    f.write('\n\n'.join(content_parts[:mid_point]))
                    f.write('\n\n')
                    
                    # 如果有第二张图片，插入到中间
                    if len(images) > 1:
                        f.write(f"![配图]({images[1]})\n\n")
                    
                    # 写入后半部分
                    f.write('\n\n'.join(content_parts[mid_point:]))
                    
                    # 如果有第三张图片，插入到末尾
                    if len(images) > 2:
                        f.write(f"\n\n![配图]({images[2]})")
                    
                    # 写入标签
                    if tags:
                        f.write("\n\n---\n")
                        f.write("\n".join([f"#{tag}" for tag in tags]))
                print(f"\n✅ 小红书版本已保存至: {xiaohongshu_file}")
                return [original_file, organized_file, xiaohongshu_file]
            except Exception as e:
                print(f"⚠️ 生成小红书版本失败: {str(e)}")
                import traceback
                print(f"错误详情:\n{traceback.format_exc()}")
            
            print(f"\n✅ 笔记已保存至: {original_file}")
            print(f"✅ 整理版内容已保存至: {organized_file}")
            return [original_file, organized_file]
            
        except Exception as e:
            print(f"⚠️ 处理视频时出错: {str(e)}")
            return []
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def process_json_file(self, input_file: str) -> None:
        """处理JSON文件，生成优化后的笔记
        
        Args:
            input_file (str): 输入的JSON文件路径
        """
        try:
            # 读取并解析JSON文件
            with open(input_file, 'r', encoding='utf-8') as f:
                video_list = json.load(f)
                
            # 验证JSON结构
            if not isinstance(video_list, list):
                raise ValueError("JSON文件必须包含一个视频信息列表")
                
            # 验证每个视频信息的结构
            for video in video_list:
                if not all(key in video for key in ['video_path', 'title', 'content']):
                    raise ValueError("每个视频信息必须包含 'video_path', 'title', 'content' 字段")
            
            print(f"✅ 成功读取 {len(video_list)} 个视频信息")
        
            # 处理每个视频链接
            for i, video in enumerate(video_list, 1):
                print(f"处理第 {i}/{len(video_list)} 个视频: {video['video_path']}\n")
                self.process_video(video['video_path'], video['title'], video['content'])
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析错误: {str(e)}")
            raise
        except Exception as e:
            print(f"⚠️ 处理JSON文件时出错: {str(e)}")
            raise

    def process_markdown_file(self, input_file: str) -> None:
        """处理markdown文件，生成优化后的笔记
        
        Args:
            input_file (str): 输入的markdown文件路径
        """
        try:
            # 读取markdown文件
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取视频链接
            video_links = re.findall(r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|bilibili\.com/video/|douyin\.com/video/)[^\s\)]+', content)
            
            if not video_links:
                print("未在markdown文件中找到视频链接")
                return
                
            print(f"找到 {len(video_links)} 个视频链接，开始处理...\n")
            
            # 处理每个视频链接
            for i, url in enumerate(video_links, 1):
                print(f"处理第 {i}/{len(video_links)} 个视频: {url}\n")
                self.process_video(url)
                
        except Exception as e:
            print(f"处理markdown文件时出错: {str(e)}")
            raise

def extract_urls_from_text(text: str) -> list:
    """
    从文本中提取所有有效的URL
    支持的URL格式：
    - 视频平台URL (YouTube, Bilibili, 抖音等)
    - 包含http://或https://的标准URL
    - 短链接URL (如t.co等)
    
    Args:
        text: 输入文本
        
    Returns:
        list: 提取到的有效URL列表
    """
    # URL正则模式
    url_patterns = [
        # 标准URL
        r'https?://[^\s<>\[\]"\']+[^\s<>\[\]"\'.,]',
        # 短链接
        r'https?://[a-zA-Z0-9]+\.[a-zA-Z]{2,3}/[^\s<>\[\]"\']+',
        # Bilibili
        r'BV[a-zA-Z0-9]{10}',
        # 抖音分享链接
        r'v\.douyin\.com/[a-zA-Z0-9]+',
    ]
    
    urls = []
    for pattern in url_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            url = match.group()
            # 对于不完整的BV号，添加完整的bilibili前缀
            if url.startswith('BV'):
                url = f'https://www.bilibili.com/video/{url}'
            urls.append(url)
    
    # 去重并保持顺序
    seen = set()
    return [url for url in urls if not (url in seen or seen.add(url))]

if __name__ == '__main__':
    import sys, os, re
    import argparse
    
    parser = argparse.ArgumentParser(description='视频笔记生成器')
    parser.add_argument('input', help='输入源：视频URL、包含URL的文件或markdown文件')
    parser.add_argument('--xiaohongshu', action='store_true', help='生成小红书风格的笔记')
    args = parser.parse_args()
    
    generator = VideoNoteGenerator()
    
    if os.path.exists(args.input):
        # 读取文件内容
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # 尝试使用gbk编码
                with open(args.input, 'r', encoding='gbk') as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ 无法读取文件: {str(e)}")
                sys.exit(1)
        
        # 如果是markdown文件，直接处理
        if args.input.endswith('.md'):
            print(f"📝 处理Markdown文件: {args.input}")
            generator.process_markdown_file(args.input)
        elif args.input.endswith('.json'):
            print(f"📝 处理JSON文件: {args.input}")
            generator.process_json_file(args.input)
        else:
            # 从文件内容中提取URL
            urls = extract_urls_from_text(content)
            
            if not urls:
                print("⚠️ 未在文件中找到有效的URL")
                sys.exit(1)
            
            print(f"📋 从文件中找到 {len(urls)} 个URL:")
            for i, url in enumerate(urls, 1):
                print(f"  {i}. {url}")
            
            print("\n开始处理URL...")
            for i, url in enumerate(urls, 1):
                print(f"\n处理第 {i}/{len(urls)} 个URL: {url}")
                try:
                    generator.process_video(url)
                except Exception as e:
                    print(f"⚠️ 处理URL时出错：{str(e)}")
                    continue
    else:
        # 检查是否是有效的URL
        if not args.input.startswith(('http://', 'https://')):
            print("⚠️ 错误：请输入有效的URL、包含URL的文件或markdown文件路径")
            print("\n使用示例：")
            print("1. 处理单个视频：")
            print("   python video_note_generator.py https://example.com/video")
            print("\n2. 处理包含URL的文件：")
            print("   python video_note_generator.py urls.txt")
            print("   - 文件中的URL可以是任意格式，每行一个或多个")
            print("   - 支持带有其他文字的行")
            print("   - 支持使用#注释")
            print("\n3. 处理Markdown文件：")
            print("   python video_note_generator.py notes.md")
            sys.exit(1)
        
        # 处理单个URL
        try:
            print(f"🎥 处理视频URL: {args.input}")
            generator.process_video(args.input)
        except Exception as e:
            print(f"⚠️ 处理URL时出错：{str(e)}")
            sys.exit(1)