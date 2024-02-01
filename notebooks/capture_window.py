import time

import Quartz.CoreGraphics as CG
import cv2
import numpy as np
import pyautogui


def get_window_info(window_title):
    windows = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID)
    for window in windows:
        if str(window.get('kCGWindowName')) == window_title:
            return window.get('kCGWindowBounds')
    return None


# 获取游戏窗口信息
game_window_title = 'Binding of Isaac: Repentance'
window_info = get_window_info(game_window_title)
if not window_info:
    print("找不到游戏窗口")
    exit()
x, y, width, height = window_info.get('X'), window_info.get('Y'), window_info.get('Width'), window_info.get('Height')
fps = 30  # 换成你的显示器刷新率
# 定义编码器，创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter('录制.png', fourcc, fps, (int(width), int(height)))
cv2.namedWindow('Game Capture')
cv2.moveWindow('Game Capture', 100, 100)  # 设置窗口位置为 (100, 100)
# 在主循环中获取游戏窗口截图并保存
print("(x, y, width, height)", (x, y, width, height))
while True:
    start_time = time.time()  # 记录开始时间
    try:
        # 使用 pyautogui 截图，并将 PIL Image 转换为 OpenCV 格式
        now = time.time()
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        print("spend", time.time() - start_time)
        if cv2.waitKey(1) == ord('q'):  # 如果用户按下 'q' 键，退出
            break
    except Exception as e:
        print(e)

    delay = 1.0 / fps - (time.time() - start_time)  # 就完全没有时间等待了， 这个状态下。 就是前面的处理太耗时间了
    print("delay", delay)
    if delay > 0:
        time.sleep(delay)
    break
out.release()
cv2.destroyAllWindows()
