import glob

import streamlit as st
from PIL import Image

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


# Function to read the caption from a file
def read_caption(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        return lines


# Function to increment the count
def next():
    st.session_state.count += 1
    st.session_state.count %= len(st.session_state.filtered_captions)


# Function to decrement the count
def prev():
    st.session_state.count -= 1
    st.session_state.count %= len(st.session_state.filtered_captions)


# Function to filter captions and images based on the category
def filter_content(category):
    captions = glob.glob(f"captions/{category}/*.txt")
    captions_zh = glob.glob(f"captions_zh/{category}/*.txt")
    images = glob.glob(f"images/{category}/*.jpg")
    return sorted(captions), sorted(captions_zh), sorted(images)


def single_image_preview():
    # Initialize a session state variable to keep track of the count and the category
    if "count" not in st.session_state:
        st.session_state.count = 0
    if "category" not in st.session_state:
        st.session_state.category = "live_subject"  # Default category
        st.session_state.filtered_captions, st.session_state.filtered_captions_zh, st.session_state.filtered_images = filter_content(st.session_state.category)

    # Dropdown to select the category
    category = st.selectbox(
        "Choose a category",
        ("live_subject/animal", "live_subject/human", "object", "style"),
        index=0,
        on_change=filter_content,
        args=(st.session_state.category,),
    )

    # Update the filtered captions and images if the category changes
    if category != st.session_state.category:
        st.session_state.category = category
        st.session_state.filtered_captions, st.session_state.filtered_captions_zh, st.session_state.filtered_images = filter_content(category)
        st.session_state.count = 0  # Reset count whenever the category changes

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Previous", on_click=prev)
    with col2:
        st.button("Next", on_click=next)

    # Progress bar
    st.progress(st.session_state.count / len(st.session_state.filtered_captions))

    # Display the current caption and image
    caption = read_caption(st.session_state.filtered_captions[st.session_state.count])
    caption_zh = read_caption(st.session_state.filtered_captions_zh[st.session_state.count])
    col_1, col_2, col_3 = st.columns(3)
    col_1.image(Image.open(st.session_state.filtered_images[st.session_state.count]))
    col_2.write("\n".join(caption))
    col_3.write("\n".join(caption_zh))


def display_all_images(category):
    # Display all images in the category
    images = glob.glob(f"images/{category}/*.jpg")
    for image in images:
        st.image(Image.open(image), caption=image, use_column_width=True)


def all_images_preview():

    # Dropdown to select the category
    category = st.selectbox(
        "Choose a category",
        ("live_subject/animal", "live_subject/human", "object", "style"),
        index=0,
    )

    captions, captions_zh, images = filter_content(category)
    cols = st.columns(5)

    for i in range(0, len(images), 5):
        for j in range(5):
            if i + j < len(images):
                with open(captions[i + j], "r") as file:
                    caption = file.readlines()
                subject = caption[0].strip()
                with open(captions_zh[i + j], "r") as file:
                    caption_zh = file.readlines()
                subject_zh = caption_zh[0].strip()
                cols[j].image(Image.open(images[i + j]), caption=f"{subject}/{subject_zh}", use_column_width=True)


page_names_to_funcs = {"Single Image & Captions": single_image_preview, "All Images": all_images_preview}

demo_name = st.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
