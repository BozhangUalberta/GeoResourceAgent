
# def online_train():
#     return model


def productivity_opt(
        userID: Annotated[str, InjectedState("userID")],
        runnable: Annotated[bool, "Does the trained model exist?"],
        target_mesh: Annotated[Optional[list[int]], "The output mesh size for the interpolated model, e.g., [20, 20]"],
):
    """
    function is used to predict the productivity of the region
    the productivity map will be presented
    """

 

    
    # Prepare the output in JSON format
    output_results = {
        "best location":f"Longitude:{long}, latitude:{lati}",
        "image_url": f"Image URL: {image_url}"
    }
    return json.dumps(output_results)
