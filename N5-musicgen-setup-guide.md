# MusicGen 本地部署教程 (RTX 3090)

## 🎯 你的配置
- **GPU**: NVIDIA RTX 3090 (24GB GDDR6X)
- **推荐模型**: `musicgen-medium` 或 `musicgen-large`
- **可生成时长**: 30 秒 - 4 分钟（分段可更长）

---

## 📦 一、环境准备

### 1.1 检查驱动
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

确保 CUDA 版本 ≥ 11.8，如需要更新：
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install nvidia-driver-535 nvidia-cuda-toolkit
```

### 1.2 创建虚拟环境（推荐）
```bash
# 创建 Python 3.10 虚拟环境
python3.10 -m venv musicgen-env
source musicgen-env/bin/activate

# 或使用 conda
conda create -n musicgen python=3.10
conda activate musicgen
```

### 1.3 安装 PyTorch (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1.4 安装 Audiocraft
```bash
# 方式 A: pip 直接安装（推荐）
pip install -U audiocraft

# 方式 B: 源码安装（可定制）
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
```

### 1.5 验证安装
```bash
python -c "from audiocraft.models import MusicGen; print('OK')"
```

---

## 🎹 二、Web UI 启动（推荐新手）

Audiocraft 自带 Gradio 界面，无需写代码：

```bash
# 启动 Web UI
python -m audiocraft.gradio

# 或指定端口
python -m audiocraft.gradio --port 7860
```

访问：http://localhost:7860

**界面功能**：
- 文本提示词输入
- 模型选择（small/medium/large）
- 生成时长设置
- 实时试听 & 下载

---

## 💻 三、Python 脚本生成（进阶）

### 3.1 基础生成脚本

创建 `generate_music.py`：

```python
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time

# 检查 GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 加载模型（3090 推荐 medium 或 large）
print("Loading model...")
model = MusicGen.get_pretrained('facebook/musicgen-medium')
# model = MusicGen.get_pretrained('facebook/musicgen-large')  # 更高质量，更慢

# 设置生成参数
model.set_generation_params(duration=30)  # 30 秒
model = model.cuda()  # 移到 GPU

# 提示词（英伦摇滚示例）
descriptions = [
    'britpop rock with jangly electric guitars, melodic bass, steady drums, upbeat tempo, 120 BPM, 90s British indie style',
    'light rock, British indie, acoustic and electric guitars, nostalgic mood, male vocals',
]

print("Generating...")
start = time.time()
wav = model.generate(descriptions)
print(f"Generation time: {time.time() - start:.2f}s")

# 保存
for idx, one_wav in enumerate(wav):
    audio_write(f'output_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
    print(f"Saved: output_{idx}.wav")
```

运行：
```bash
python generate_music.py
```

### 3.2 分段生成长歌曲

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

model = MusicGen.get_pretrained('facebook/musicgen-medium')
model = model.cuda()

# 生成主歌部分（30 秒）
model.set_generation_params(duration=30)
wav1 = model.generate(['britpop verse, jangly guitars, building up'])

# 生成副歌部分（30 秒）
wav2 = model.generate(['britpop chorus, anthemic, energetic, full band'])

# 生成桥段（15 秒）
model.set_generation_params(duration=15)
wav3 = model.generate(['bridge, softer, introspective'])

# 保存
audio_write('verse', wav1[0].cpu(), model.sample_rate, strategy="loudness")
audio_write('chorus', wav2[0].cpu(), model.sample_rate, strategy="loudness")
audio_write('bridge', wav3[0].cpu(), model.sample_rate, strategy="loudness")

# 后续在 DAW 中拼接
```

---

## ⚙️ 四、参数优化指南

### 4.1 模型选择

| 模型 | VRAM 占用 | 音质 | 速度 | 推荐用途 |
|------|----------|------|------|---------|
| `musicgen-small` | ~6GB | ⭐⭐⭐ | 快 | 测试/快速迭代 |
| `musicgen-medium` | ~12GB | ⭐⭐⭐⭐ | 中等 | **日常使用（推荐）** |
| `musicgen-large` | ~18GB | ⭐⭐⭐⭐⭐ | 慢 | 最终成品 |
| `musicgen-melody` | ~12GB | ⭐⭐⭐⭐ | 中等 | 有旋律参考时 |

**3090 建议**：日常用 `medium`，最终成品用 `large`

### 4.2 生成时长

```python
# 短片段（8-15 秒）- 快速测试
model.set_generation_params(duration=8)

# 标准片段（30 秒）- 平衡质量/速度
model.set_generation_params(duration=30)

# 长片段（60-120 秒）- 需要更多 VRAM
model.set_generation_params(duration=60)

# 注意：超过 2 分钟建议分段生成
```

### 4.3 温度参数（控制随机性）

```python
# 低温度（0.5-0.7）- 更稳定、可预测
model.set_generation_params(temperature=0.7)

# 中温度（0.8-1.0）- 平衡
model.set_generation_params(temperature=1.0)

# 高温度（1.1-1.5）- 更创意、但可能不稳定
model.set_generation_params(temperature=1.2)
```

### 4.4 Top-K / Top-P 采样

```python
# 更聚焦的采样
model.set_generation_params(top_k=250, top_p=0.95)

# 更多样化的采样
model.set_generation_params(top_k=500, top_p=0.99)
```

---

## 🎸 五、英伦摇滚提示词库

### 5.1 基础风格

```
# 经典 Britpop
'britpop rock, jangly electric guitars, melodic bass, steady drums, 120 BPM, 90s British indie, Oasis style'

# 抒情向
'britpop ballad, acoustic guitar, piano, emotional male vocals, nostalgic, 90 BPM, Blur style'

# 活力向
'upbeat britpop, energetic drums, bright guitars, anthemic chorus, 128 BPM, The Verve style'
```

### 5.2 乐器细节

```
# 吉他
'jangly Rickenbacker guitar tone, chorus effect, arpeggiated chords'

# 贝斯
'melodic bass guitar, prominent in mix, Paul McCartney style'

# 鼓
'live drum kit, room reverb, steady rock beat, fill transitions'

# 键盘
'Hammond organ, Wurlitzer electric piano, subtle strings'
```

### 5.3 制作风格

```
'90s British indie production, warm analog sound, slightly lo-fi, tape saturation'
'modern britpop production, clean mix, radio-ready, wide stereo'
```

### 5.4 完整提示词示例

```python
prompts = [
    # 示例 A: 励志摇滚
    'britpop rock anthem, jangly electric guitars, driving bass, powerful drums, '
    'male british vocals, uplifting chorus, 120 BPM, Oasis Don\'t Look Back in Anger style, '
    '90s production, warm analog sound',
    
    # 示例 B: 抒情民谣
    'britpop ballad, acoustic guitar fingerpicking, piano, soft strings, '
    'emotional male vocals, nostalgic mood, 85 BPM, Blur Tender style, '
    'intimate production, close-mic\'d vocals',
    
    # 示例 C: 独立摇滚
    'british indie rock, bright electric guitars, melodic bass, energetic drums, '
    'female vocals british accent, dreamy atmosphere, 115 BPM, The Verve Bitter Sweet style, '
    'reverb-heavy production, spacious mix',
]
```

---

## 🔧 六、常见问题解决

### 6.1 CUDA 内存不足
```bash
# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 或在脚本中
torch.cuda.empty_cache()

# 使用 smaller 模型
model = MusicGen.get_pretrained('facebook/musicgen-small')
```

### 6.2 生成速度慢
```python
# 使用半精度（节省 VRAM，加快速度）
model = MusicGen.get_pretrained('facebook/musicgen-medium')
model = model.half()  # FP16
model = model.cuda()

# 减少生成时长
model.set_generation_params(duration=15)  # 先测试短片段
```

### 6.3 音质不佳
- 使用 `large` 模型
- 增加生成时长（分段后拼接）
- 调整温度参数（0.7-0.9 通常最好）
- 提示词更具体（加入乐器、BPM、参考艺人）

---

## 🎼 七、后期工作流

### 7.1 导出格式
```python
# 默认导出 WAV（无损）
audio_write('output', wav[0].cpu(), model.sample_rate, strategy="loudness")

# 手动导出为 MP3
import torchaudio
torchaudio.save('output.mp3', wav[0].cpu(), model.sample_rate, encoding='MP3')
```

### 7.2 在 DAW 中处理
1. 导入生成的 WAV 文件
2.  EQ 调整（切低频噪音，增强高频）
3.  压缩（让人声/吉他更突出）
4.  混响（增加空间感）
5.  母带处理（Landr 或手动）

### 7.3 拼接多段
用 Audacity（免费）或 Reaper：
1. 导入所有分段
2. 按结构排列（主歌 - 副歌 - 桥段）
3. 交叉淡化过渡
4. 导出完整歌曲

---

## 📊 八、性能参考（RTX 3090）

| 模型 | 时长 | 生成时间 | VRAM 占用 |
|------|------|---------|----------|
| small | 15 秒 | ~10 秒 | 6GB |
| small | 30 秒 | ~20 秒 | 8GB |
| medium | 15 秒 | ~20 秒 | 12GB |
| medium | 30 秒 | ~40 秒 | 14GB |
| large | 15 秒 | ~45 秒 | 18GB |
| large | 30 秒 | ~90 秒 | 20GB |

*实际时间因系统而异*

---

## 🚀 九、快速开始清单

```bash
# 1. 创建虚拟环境
python3.10 -m venv musicgen-env
source musicgen-env/bin/activate

# 2. 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 Audiocraft
pip install -U audiocraft

# 4. 启动 Web UI
python -m audiocraft.gradio

# 5. 访问 http://localhost:7860 开始创作！
```

---

## 💡 十、进阶技巧

### 10.1 旋律引导生成
```python
import torchaudio

# 加载参考旋律
melody, sr = torchaudio.load('reference_melody.mp3')

# 用旋律引导生成
wav = model.generate_with_chroma(
    descriptions=['britpop rock with jangly guitars'],
    melody=melody[None].expand(1, -1, -1),
    melody_sr=sr
)
```

### 10.2 批量生成
```python
prompts = [
    'britpop rock, upbeat, 120 BPM',
    'britpop ballad, slow, 80 BPM',
    'britpop indie, medium tempo, 100 BPM',
]

for i, prompt in enumerate(prompts):
    wav = model.generate([prompt])
    audio_write(f'batch_{i}', wav[0].cpu(), model.sample_rate, strategy="loudness")
    print(f'Generated: batch_{i}')
```

### 10.3 条件生成（延续已有音频）
```python
# 加载已有音频作为前奏
continuation, sr = torchaudio.load('intro.wav')

# 延续生成
model.set_generation_params(duration=30)
wav = model.generate(
    descriptions=['continue with full band, verse section'],
    continuation=continuation[None].cuda()
)
```

---

## 📚 资源链接

- **官方 GitHub**: https://github.com/facebookresearch/audiocraft
- **Hugging Face 模型**: https://huggingface.co/facebook/musicgen-medium
- **在线 Demo**: https://huggingface.co/spaces/facebook/MusicGen
- **Discord 社区**: https://discord.gg/audiocraft

---

**祝你创作顺利！🎵**

有问题随时问～
