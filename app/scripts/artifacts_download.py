from app.services.artifact_service import download_required_artifacts


def main():
    download_required_artifacts()
    print("All artifacts downloaded successfully!")


if __name__ == "__main__":
    main()
