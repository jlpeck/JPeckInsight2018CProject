
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input

import pandas as pd
import numpy as np

import spacy
import ftfy
from pack64 import unpack64

# load nlp model
nlpmed = spacy.load('en_core_web_md')

# HOW MANY RESULTS RETURNED TO USER
NUMBER_OF_HITS = 3

# read in dataframe
df = pd.read_csv("MVPfrontenddata.csv")

# app splashscreen image
image_path = 'assets/rawpixel-718387-unsplash.jpg'


def norm_vector(x):
    """
    Arguments:
    x (vector): a vector of floats
    
    Returns:
    normedvector (vector): a vector standardized to length 1
    """
    normedvector = x/np.sqrt(x@x)
    return normedvector


def unpack_vectors(dataframecolumn):
    """
    Arguments: 
    dataframecolumn (pandas dataframe column): 
        single column of a pandas dataframe containing pack64'd document vectors
    
    Returns: 
    newarray (numpy array): array of unpacked document vectors
    """
    newarray = np.asarray([unpack64(x) for x in dataframecolumn])
    return newarray


def process_vector_comparison(jobvectorarray, userinputstring):
    """
    Arguments: 
    jobvectorarray: array of vectorized job descriptions
    userinputstring: input text string from user
    
    Returns: 
    result (array): sorted, truncated (top ten) array that is a vector of 
    cosine similarity scores and a vector of the corresponding indeces from the 
    original dataframe so the original text of the job description may be retrieved.
    """
    
    meanvector = np.mean(jobvectorarray,axis=0)
    
    inputvector = np.mean(np.asarray([y.vector for y in nlpmed(ftfy.fix_text(userinputstring)) if not y.is_stop]), axis = 0)
    
    processed_input_vector =  norm_vector(inputvector) - meanvector
    
    newarray = np.asarray([norm_vector(x - meanvector) for x in jobvectorarray])
    
    scoresarray = pd.DataFrame(newarray@processed_input_vector)
    
    scoresarray['marker'] = scoresarray.index
    scoresarray.columns = ['scores', 'marker']
    result = scoresarray.sort_values(['scores'], ascending = False).head(10)
        
    return result
    

def convert_scores_to_indicators(scoresarray):
    """
    Arguments:
    scoresarray (array): sorted array of cosine similarity scores
    
    Returns:
    indicator_list (list): list of strings ("Good Match", "Ok Match", "Bad Match")
    """
    indicator_list = []
    for score in scoresarray['scores']:
        print(score)
        if score > 0.2:
            indicator_list.append("Great Match")
        elif (score > 0) & (score <= 0.2):
            indicator_list.append("OK Match")
        elif (score > -0.2) & (score <= 0):
            indicator_list.append("Poor Match")
        elif (score < -0.2) or np.isnan(score):
            indicator_list.append("Bad Match")
        else:
            indicator_list.append("Bad Match")
    return indicator_list


def serve_user_recs(rec_df, orig_df,numberofrows):
    """
    Argument: 
    rec_df (dataframe): an array of scores corresponding description indeces
    orig_df (dataframe): dataframe containing scraped Indeed.com job descriptions
    numberofrows (int): how many hits would you like to serve users?
    
    Returns:
    results_df (dataframe): short dataframe of results to serve users
    """ 
    
    markers = rec_df['marker'].iloc[0:numberofrows]
    variables = ['title', 'company', 'description', 'url']
    dfselections = pd.DataFrame(orig_df.iloc[markers][variables])
    dfselections['description'] = [x[0:180] 
        for x in dfselections['description'] 
        if len(x)>= 180]
    
    # retained in case numeric scores, not text indicators, are required
    #formattedscore = [ '{:.2f}'.format(x) for x in rec_df['scores'][0:numberofrows] ] 
    
    translatedscore = convert_scores_to_indicators(rec_df)[0:numberofrows]    
    
    results_df = pd.DataFrame({'Job_Title': dfselections['title'], 
                               'Company': dfselections['company'], 
                               'Job_Description': dfselections['description'],
                               'Link':dfselections['url'],
                               'Score': translatedscore})   
    return results_df            
            

def do_NLP_to_stuff(jobs_df, user_input, numberofhits):
    """
    Input: 
    jobs_df (dataframe): dataframe of job post data from Indeed.com in which 
        job descriptions already vectorized and serialized for 
        storage in a df column.
    user_input (string): text string of user input describing a job
    numberofhits (int): number of results to return
    
    Unpacks vectors from original dataframe.
    Processes those vectors by de-meaning and matrix multiplication
    with user input vector.
    Serves top match by cosine similarity score for display on webapp.
    
    Returns
    results_df (dataframe): pandas dataframe of user results
    """
    job_vectors = unpack_vectors(df['packed_jobvectors'])
    cosine_scores = process_vector_comparison(job_vectors, user_input)
    results_df = serve_user_recs(cosine_scores, jobs_df, numberofhits)
    
    return results_df


def make_table_cell(dataframe, index, column):
    """
    Arguments:
    dataframe (dataframe): results dataframe to serve webapp
    index (int): row index of dataframe
    column (string): name of column
        
    Format the links column as clickable links
    with an href by passing through any table element unchanged unless
    it comes from the link column. 
    
    Returns link object or passes table element unchanged.
    """
    if column == "Link":
        return html.A(children = dataframe.iloc[index][column], href= dataframe.iloc[index][column])
    else:
        return dataframe.iloc[index][column]
        

def generate_table(results_df, max_rows=10):
    """
    Arguments:
    results_df (dataframe): 
    
    Converts pandas dataframe to html table to serve user results in webapp.
    return: html table object   
    """
    
    return html.Table(
        # Header
        [html.Tr([html.Th(col, style={'textAlign': 'center', 'border-collapse': 'separate', 'border-spacing': '10px'}) 
        for col in results_df.columns],  style={'textAlign': 'center', 'border-collapse': 'separate', 'border-spacing': '10px'})] +

        # Body
        [html.Tr([html.Td(make_table_cell(results_df, i, col)) for col in results_df.columns], 
        style={'textAlign': 'center', 'border-collapse': 'separate', 'border-spacing': '10px'}) 
                    for i in range(min(len(results_df), max_rows))], 
        style={'textAlign': 'center', 'border-collapse': 'separate', 'border-spacing': '10px'})


app = dash.Dash(__name__)

app.layout = html.Div([
    
    # Nav Bar
    html.Nav([
        html.Div([
            html.A("NextNiche", className = "navbar-brand js-scroll-trigger", href = "#page-top"),

                    html.A("About Me", className = "navbar-text js-scroll-trigger", href="#about-me", style={"float": "right"}),
                    html.A("Demo", className = "navbar-text js-scroll-trigger", href="#demo", style={"float": "right"}),                        
                    html.A("Home", className = "navbar-text js-scroll-trigger", href="#page-top", style={"float": "right"})                        
                           ], className = "container")
        ], className = "navbar navbar-default navbar-static-top", style = {"overflow":"hidden", "position": "fixed", "top": "0", "width": "100%"}),
    
    # For proper navigation back to the top, leave this div between Nav and Jumbotron
    html.Div([], id = "page-top"),
    
    # Jumbotron
    html.Div([
        html.Div([
                html.H1("NextNiche", className = "display-4", style = { "margin-top": "70px",'color' : "white", "vertical-align": "middle"}),
                html.H4("An NLP-Based Job-Discovery Engine", style = {'color' : "white", "vertical-align": "middle"})
            ], className = "container text-center", style={}),
    ], className = "jumbotron jumbotron-fluid", style={"align-items": "center", "height": "100vh", "margin": "0px", "padding": "0px", 'background-size':'cover', 'background-image': 'url({})'.format(image_path)}),
 

    # This Div Contains All Rows
    html.Div([


    # For proper navigation back to the top, leave this div between Nav and Jumbotron
    html.Div([], id = "demo"),
    
    
    # ROW 1    
    html.Div( className = "row", style = {'margin-top': '60px'}),
        

    # ROW 2          
    html.Div( 
            [html.Label('Someday, I want to find a role where...')
    ], style = {}, className = "row"),
    
    
    # ROW 3
    html.Div([
        dcc.Textarea(id='job-wishes',
                     placeholder='Enter text',
                    value='enter text',
                    style={'width': '100%'}
                ),
                html.Button(id='submit-button', type='submit', children='Submit',
                            n_clicks=0),
            ], className = "row"),
                            

    #ROW 4
    html.Div([html.Div(id='output_div' )], className = "row"),

    
    # ROW 5
    html.Div([       
    ], className = "row"
    ),
            
            
    # ROW 6
    html.Div([
            html.H3("About Me", className = "text-center"),
            html.P("Hi. I'm Jessica Peck.", className = "text-center"),
            html.P("I'm a PhD in Economics and a Fellow at Insight Data Science.", className = "text-center"),
            html.P("My research focuses on the interesection of driving and health,", className = "text-center"),
            html.P("most recently studying the effects of Uber access on drunk driving in New York State.", className = "text-center")
    ], className = "row", id = "about-me", style = {"align-items": "center", "margin-top": "300px" }
    )
    
    ], className = "container")
                    
], className = "col-12")
          
                
    
    
# Listens for change in n_clicks, takes job-wishes value, puts in output field
@app.callback(Output('output_div','children'),
              [Input('submit-button','n_clicks')],
              [State('job-wishes', 'value')]
        )
def getJobInfo(n_clicks, input1):
    if input1 == "enter text":
        return ""
    else: 
        result_df = do_NLP_to_stuff(df, input1, NUMBER_OF_HITS)
        return generate_table(result_df)


if __name__ == '__main__':
    app.run_server(debug=True)
    


