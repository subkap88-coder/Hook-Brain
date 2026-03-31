if __name__ == '__main__':
    from tribev2 import TribeModel

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    df = model.get_events_dataframe(text_path="hook.txt")
    preds, segments = model.predict(events=df)

    print("Output shape:", preds.shape)
    print("Min activation:", preds.min())
    print("Max activation:", preds.max())
    print("Mean activation:", preds.mean())
    print("Segments:", segments)
