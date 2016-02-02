from allograph.learning_manager import save_learned_allograph as save_letter
from allograph.stroke import Stroke
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
import inspect
import argparse
parser = argparse.ArgumentParser(description='Learn a collection of drawings')
parser.add_argument('draw', action="store",
                help='The draw to be learnt')



fileName = inspect.getsourcefile(Stroke)
installDirectory = fileName.split('/lib')[0]
datasetDirectory = installDirectory + '/share/allograph/letter_model_datasets/bad_letters'
robotDirectory = installDirectory + '/share/allograph/robot_tries/start'
letter = Stroke()

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
        letter.append(touch.x, touch.y)

    def on_touch_up(self, touch):
        global letter
        if touch.is_double_tap:
            print letter.get_len()
            letter.downsampleShape(70)
            print letter.get_len()
            save_letter(robotDirectory, letter_name, letter)
            letter.reset()
            print letter.get_len()
            self.canvas.clear()
        
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
