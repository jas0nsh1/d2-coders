import color_analysis
import numpy as np

def video_extraction(video, interval):
    arrayImages = video_splice_to_discrete(video, interval)
    arrayDominantColors = []
    for i in arrayImages:
        arrayDominantColors.append(extract_dominant_colors(arrayImages[i]))

    return np.array(arrayDominantColors)


def video_splice_to_discrete(video, interval):
    return []


def color_variance(arrayDominantColors):



def match_with_company_vibe(arrayDominantColors, companyDominantColors):




