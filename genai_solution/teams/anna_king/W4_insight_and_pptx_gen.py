import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
import json
import boto3
from botocore.exceptions import ClientError
import os

@st.cache_data
def load_total_data():
    return pd.read_csv(DATA_FOLDER + "monthly_totals.csv")

@st.cache_data
def load_product_data():
    return pd.read_csv(DATA_FOLDER + "monthly_product_totals.csv")

DATA_FOLDER = os.getcwd().split("genai_solution")[0] + "genai_solution\\data\\"


total = load_total_data()
product_data = load_product_data()

month_list = sorted(list(product_data["order_month"].unique()), reverse=True)


def get_output_aws(prompt):
    client = boto3.client("bedrock-runtime", region_name="eu-west-1")

    model_id = "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    request = json.dumps(native_request)

    try:
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    return model_response["content"][0]["text"]



def gen_insight(data, data_type):
    data_loaded = DataFrameLoader(data, page_content_column="product_category_name_english").load()
    prompt = "The following is the data on the " + data_type + " product categories. Highlight three insights from this data in bullet format. Data: "  + str(data_loaded)
    return get_output_aws(prompt)

@st.cache_data(show_spinner="Generating LLM insights...")
def llm_query_df(month):
    product_data = load_product_data()
    one_month_product_data = product_data[product_data["order_month"] == month]

    # snip_1 = one_month_product_data.sort_values("total_price", ascending=False).head()
    # snip_2 = one_month_product_data.sort_values("unique_orders", ascending=False).head()
    # snip_3 = one_month_product_data.sort_values("MoM_total_price", ascending=False).head()
    # snip_4 = one_month_product_data.sort_values("MoM_total_price", ascending=True).head()
    # snip_5 = one_month_product_data.sort_values("YoY_total_price", ascending=False).head()
    # snip_6 = one_month_product_data.sort_values("YoY_total_price", ascending=True).head()

    # highest_earners = gen_insight(snip_1, "highest earning")
    # most_ordered = gen_insight(snip_2, "most ordered")
    # high_monthly_growth = gen_insight(snip_3, "fastest growing (monthly)")
    # high_yearly_growth = gen_insight(snip_5, "fastest growing (yearly)")

    # i1 = [i for i in highest_earners.split("\n")[1:] if len(i) > 0]
    # i2 = [i for i in most_ordered.split("\n")[1:] if len(i) > 0]
    # i3 = [i for i in high_monthly_growth.split("\n")[1:] if len(i) > 0]
    # i4 = [i for i in high_yearly_growth.split("\n")[1:] if len(i) > 0]

    insights = ['Computers and accessories is the highest-earning product category in January 2018, with total sales of $44,707.58.',
                'Watches and gifts category shows the highest year-over-year growth in unique orders (16.09%) and unique customers (15.91%), despite having the lowest total sales among the top categories.',
                'The bed, bath, and table category had the highest number of unique orders (412) and unique customers (408) in January 2018.',
                'Computers accessories showed the strongest year-over-year growth in January 2018, with a 10.39% increase in total price, 12.86% increase in total freight, and 11% increase in unique orders compared to January 2017.',
                'Despite having fewer unique orders and customers than some other categories, "computers_accessories" had the highest total price ($44,707.58) in January 2018.',
                'Construction tools and lights category showed the highest month-over-month growth in total price, with a 12.27% increase.',
                'The art category experienced the second-highest growth in total price, with a 10.89% month-over-month increase, and also saw a significant increase in unique orders and customers (2.33 times more than the previous month).',
                'The furniture bedroom category had the highest growth in unique orders and customers, increasing by 4 times compared to the previous month, despite having a lower growth rate in total price (3.94%).',
                'Luggage accessories show the highest year-over-year (YoY) growth in total price at 24.96%.',
                'Musical instruments have the highest YoY growth in unique orders and unique customers at 35%.']

    return insights, one_month_product_data


st.title("PowerPoint Generation")
st.subheader("Performance Review Demo")
st.write(
    """This app is a proof of concept for the use of Generative AI to extract insights from performance data, and generate a PowerPoint.
    """
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

data_url = "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/master/palmerpenguins/data/penguins.csv"

# df = pd.read_csv(data_url)
# 

month = st.selectbox("Select month...", options=month_list)

insights, data = llm_query_df(month)

st.dataframe(filter_dataframe(data))

st.write("All Insights:\n\n- ", str("\n\n- ".join(insights)))

options = st.multiselect(
    "Select your insights...",
    insights,
)


st.write("You selected:\n\n- ", "\n\n- ".join(options))

create = st.button("Create PowerPoint")

if create:
    from pptx import Presentation

    # Create a new presentation
    prs = Presentation()

    # Add a slide with a 'Title and Content' layout
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)

    # Set the title
    slide.shapes.title.text = "Performance Insights"

    # Access the content placeholder
    content = slide.placeholders[0]

    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.dml.color import RGBColor
    from pptx.util import Inches


    #     color='#3182bd', # color of line
    #     label="Computer Accessories" # label for legend

    #     color='#9ecae1', # color of line
    #     label="Luggage Accessories" # label for legend

    #     markerfacecolor='blue', # color of marker
    #     color='#deebf7', # color of line
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1), Inches(4),Inches(8),Inches(0.8))
    shape.shadow.inherit = False
    fill=shape.fill
    fill.solid()
    fill.fore_color.rgb=RGBColor(8, 81, 156)
    shape.line.color.rgb = RGBColor(8, 81, 156)
    shape.text= """Computers and accessories is the highest-earning product category in January 2018, with total sales of $44,707.58."""


    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1),  Inches(5),Inches(8),Inches(0.8))
    shape.shadow.inherit = False
    fill=shape.fill
    fill.solid()
    fill.fore_color.rgb=RGBColor(49, 130, 189)
    shape.line.color.rgb = RGBColor(49, 130, 189)
    shape.text= """Luggage accessories show the highest year-over-year (YoY) growth in total price at 24.96%."""

    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1), Inches(6),Inches(8),Inches(0.8))
    shape.shadow.inherit = False
    fill=shape.fill
    fill.solid()
    fill.fore_color.rgb=RGBColor(107, 174, 214)
    shape.line.color.rgb = RGBColor(107, 174, 214)
    shape.text= """Construction tools and lights category showed the highest month-over-month growth in total price, with a 12.27% increase."""
    # text_frame = content.text_frame

    # # Clear any existing paragraphs
    # text_frame.clear()

    # # Add the first bullet point
    # p1 = text_frame.paragraphs[0]
    # p1.text =  'Computers and accessories is the highest-earning product category in January 2018, with total sales of $44,707.58.'

    # # Add more bullet points
    # p2 = text_frame.add_paragraph()
    # p2.text =  """Construction tools and lights category showed the highest month-over-month growth in total price, with a 12.27% increase."""


    # p3 = text_frame.add_paragraph()
    # p3.text = 'Luggage accessories show the highest year-over-year (YoY) growth in total price at 24.96%.'

    from pptx.util import Inches, Pt
    imgpth=DATA_FOLDER + 'graph.png'
    left = Inches(1)
    top = Inches(1.5)
    pic = slide.shapes.add_picture(imgpth, left, top)


    # Save the presentation
    prs.save("Performance Insights.pptx")
    st.write("Powerpoing Created!")
    