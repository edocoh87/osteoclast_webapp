"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
import streamlit as st
from infer import predict2

# set title of app
st.title("Osteoclast Quantification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "png")
# file_up = st.file_uploader("Upload an image", type = "jpg")

if file_up is not None:
    # display image that user uploaded
    print('file_up {}'.format(file_up))
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    slices_per_axis = st.slider('How many slices per axis? (for full cultures use at least 3)', 1, 5, format="%d slices")
    print('slices_per_axis', slices_per_axis)
    st.write("You selected ", slices_per_axis, 'slices per axis.')
    # slices_per_axis = st.selectbox(
    #     'How many patches per axis? (for full cultures use at least 3)',
    #     (1, 2, 3, 4, 5))
    # st.write('You selected:', slices_per_axis)
    # if slices_per_axis:
    # slices_per_axis = st.text_input('Images are divided to grid of SxS')
    if st.button('Go'):
        # if slices_per_axis:
        slices_per_axis = int(slices_per_axis)
        st.write("Working on it (can take a while, depending on your hardware...)")
        output_image, res = predict2(image,  file_up.name, slices_per_axis)
        st.image(output_image, caption='Here is the output', use_column_width=True)

        st.write("Number of rectangles: ", res['rects'])
        st.write("Relative area: ", res['relative_area'])