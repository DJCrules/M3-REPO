def MSE(outputs, answers):
    error = 0
    for output, answer in outputs, answers:
        error+=(answer - output) ^ 2
    error/=len(outputs)
    return error

def Linear_MSE(m, )

training_data = [1, 2, 3, 4, 4, 6, 7, 8]

def Linear_Regression(training_data):
    m = (training_data[1] - training_data[0])
