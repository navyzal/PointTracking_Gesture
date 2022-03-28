from pynput.mouse import Button, Controller
import time
 
# 마우스 콘트롤러 객체
mouse = Controller()
pos = None
while True:
    if pos:
        mouse.move(10,10)
        print('이동된 마우스 포지션: {0}'.format(mouse.position))
        
    print('현재 마우스 포지션: {0}'.format(mouse.position))
    pos = mouse.position

    time.sleep(3)
# # 마우스 포지션 이동
# mouse.position = (500, 250)
 
# print('바뀐 마우스 포지션: {0}'.format(mouse.position))
 
# # 마우스 눌렀다 뗀다
# mouse.press(Button.left)
# mouse.release(Button.left)
# time.sleep(1)
 
# # 마우스 포지션 이동
# mouse.move(100, -100)
# print('바뀐 마우스 포지션: {0}'.format(mouse.position))
 
# # 더블 클릭한다
# mouse.click(Button.left, 2)