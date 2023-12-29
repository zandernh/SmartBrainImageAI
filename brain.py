from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()

prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join('mobilenet_v2-b0353104.pth'))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join('house.jpg'), result_count=5)
print('House:\n')
for each_prediction, each_probability in zip(predictions, probabilities):
    print(f'{each_prediction} : {each_probability}\n')
