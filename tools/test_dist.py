from allograph.learning_manager import save_learned_allograph as save_letter
from allograph.stroke import Stroke
import allograph.stroke as stroke
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
import inspect
import argparse
parser = argparse.ArgumentParser(description='Learn a collection of drawings')
parser.add_argument('draw', action="store",
                help='The draw to be learnt')


letter1 = Stroke()
letter2 = Stroke()
turn = True

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:

            #self.canvas.clear()
            Color(1, 1, 0)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        global lastStroke
        touch.ud['line'].points += [touch.x, touch.y]
        if turn:
            letter1.append(touch.x, touch.y)
        else:
            letter2.append(touch.x, touch.y)

    def on_touch_up(self, touch):
        global letter1
        global letter2
        global turn
        if touch.is_double_tap:
            if turn:
                letter1.downsampleShape(150)
                self.canvas.clear()
            else:
                letter2.downsampleShape(150)
                self.canvas.clear()

                # distance (letter1, letter2)
                #strokes1 = letter1.split_non_differentiable_points(1.2)
                #strokes2 = letter2.split_non_differentiable_points(1.2)
                #print len(strokes1)
                #print len(strokes2)
                #dist = stroke.compare(strokes1, strokes2)

                letter1.normalize_wrt_max()
                letter2.normalize_wrt_max()
                dist = stroke.cloud_dist(letter1,letter2)
                print dist

                letter1.reset()
                letter2.reset()
            turn = not turn
        
class UserInputCapture(App):

    def build(self):
        self.painter = MyPaintWidget()
        return self.painter

    def on_start(self):
        with self.painter.canvas:
            print(self.painter.width)
            Color(1, 1, 0)
            d = 30.
            #x = self.painter.width
            #Line(points=(x, 0, x, self.painter.height))

if __name__=="__main__":
    args = parser.parse_args()
    letter_name = args.draw

    try:
        UserInputCapture().run()
        
    except KeyboardInterrupt:
            logger.info("Bye bye")
