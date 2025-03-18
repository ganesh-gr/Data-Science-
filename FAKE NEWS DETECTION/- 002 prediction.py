import pickle

var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))


#function to run for prediction
def detecting_fake_news(var):    
    # Specify the encoding (use 'rb' for binary mode)
    with open(r"C:\Users\M.Geethasree\OneDrive\Desktop\important\Fake_News_Detection\final_model.sav", 'rb') as file:
        # Load the model
        load_model = pickle.load(file)

    # Make predictions
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return (print("The given statement is ",prediction[0]),
        print("The truth probability score is ",prob[0][1]))


if __name__ == '__main__':
    detecting_fake_news(var)
