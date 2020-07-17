---
layout: post
published: true
title:  Using Flask to Deploy a Recommendation Engine
---
## Using Flask to Deploy a Recommendation Engine

After studying data engineering, I worked on a project titled "Medcabinet", an app for new cannabis consumers who want to use cannabis as a means to battle medical conditions and ailments. I built an API with Flask to make the predictions available to the Web team. The data was sourced from [Leafly](https://www.leafly.com/). I was tasked with integrating the NLP models developed by [Geoffrey Gordon Ashbrook](https://www.linkedin.com/in/geoffrey-gordon-ashbrook/) into the Flask API and setting up the routes that would match strains to users based on search terms.

The model was developed using an adaptation of Bascilica's word embedding and Cosine Similarity.

The pickled models used for developing the API can be found in the [GitHub repository](https://github.com/dggasque/medicalcabinet_api) for this project. The Flask framework streamlines setting up APIs. I created three endpoints to allow for different search options.

As an example, the "strains" endpoint returns the top 5 search results based on desired feelings and/or flavor profile.

It only takes a few lines of code to get the endpoint up and running. Our prediction functions require importing Bascilica, Numpy, Pandas, and Scipy.

```python
from flask import Flask, request
import basilica
import numpy as np
import pandas as pd
from scipy import spatial
```

Then, all you have to do is create a Flask instance.

```python
app = Flask(__name__)
```

Now you are ready to set up your endpoint. We worked with the web team to coordinate the input and output formats. With Flask's request function, we grab the JSON blob as a python dictionary object. We then call the prediction function, using the 'input' value from the dictionary, and return the results as a JSON blob. Our Machine Learning engineer converted to JSON within the prediction function. Alternatively, we could have used Flask's jsonify function to convert our dictionary into a JSON blob.

```python

@app.route('/strains', methods=['Post'])
def strains():
    """ a route, expects json object with 1 key """

    # receive input
    lines = request.get_json(force=True)

    # get data from json
    text = lines['input']  

    # validate input (optional)
    assert isinstance(text, str)


    # predict
    output = predict(text)


    # give output to sender.
    return output

```

And the code for the prediction model:

```python
# Set Default user input
user_input = "text, Relaxed, Violet, Aroused, Creative, Happy, Energetic, Flowery, Diesel"


def predict(user_input):

    # getting data
    df = pd.read_csv('symptoms8_medcab3.csv')

    #effcts unpickling file of embedded cultivar descriptions
    unpickled_df_test = pd.read_pickle("./medembedv2.pkl")

    # Part 1
    # a function to calculate_user_text_embedding
    # to save the embedding value in session memory
    user_input_embedding = 0

    def calculate_user_text_embedding(input, user_input_embedding):

        # setting a string of two sentences for the algo to compare
        sentences = [input]

        # calculating embedding for both user_entered_text and for features
        with basilica.Connection('36a370e3-becb-99f5-93a0-a92344e78eab') as c:
            user_input_embedding = list(c.embed_sentences(sentences))

        return user_input_embedding

    # run the function to save the embedding value in session memory
    user_input_embedding = calculate_user_text_embedding(user_input, user_input_embedding)




    # part 2
    score = 0

    def score_user_input_from_stored_embedding_from_stored_values(input, score, row1, user_input_embedding):

        # obtains pre-calculated values from a pickled dataframe of arrays
        embedding_stored = unpickled_df_test.loc[row1, 0]

        # calculates the similarity of user_text vs. product description
        score = 1 - spatial.distance.cosine(embedding_stored, user_input_embedding)

        # returns a variable that can be used outside of the function
        return score



    # Part 3
    for i in range(2351):
        # calls the function to set the value of 'score'
        # which is the score of the user input
        score = score_user_input_from_stored_embedding_from_stored_values(user_input, score, i, user_input_embedding)

        #stores the score in the dataframe
        df.loc[i,'score'] = score

    # Part 4: returns all data for the top 5 results as a json obj
    df_big_json = df.sort_values(by='score', ascending=False)
    df_big_json = df_big_json.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
    df_big_json = df_big_json[:5]
    df_big_json = df_big_json.to_json(orient='columns')

    # Part 5: output
    return df_big_json

```

the other endpoints use the same process, only calling different prediction functions. The pre-trained models, along with the full application code, are available in the [GitHub repository](https://github.com/dggasque/medicalcabinet_api). The front-end application can be found [here](https://adoring-kalam-37ae7a.netlify.app/).

Additionally, a dosing information app is available here: [Potency Calculator built with Dash](https://potency-calc.herokuapp.com/), and the respective repository here: [Github Repository for Dash App](https://github.com/dggasque/Weed-Calculator).
