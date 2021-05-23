import concurrent.futures
import io
import os
import re
import sys

import requests
from PIL import Image
from bs4 import BeautifulSoup
from random_user_agent.params import SoftwareName, OperatingSystem
from random_user_agent.user_agent import UserAgent
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from tqdm import tqdm


def download_image_from(link, directory, name):
    """takes in a link to an image and saves it

    :param directory: directory to place image in
    :param link: link to image
    :param name: name to save image to
    """
    img_content = requests.get(link).content
    image_file = io.BytesIO(img_content)
    image = Image.open(image_file).convert('RGB')
    image.save(f'./{directory}/{name}.png', 'PNG', quality=100, subsampling=0)


def get_image_links(page_number, hair_color_tag):
    """gets the image links off of a page of www.animecharacterdatabase.com

    :param page_number: page number of results of these tags
    :param hair_color_tag: an integer that specifies hair color
    :return: a list of links to images
    """
    page_url = f"https://www.animecharactersdatabase.com/ux_search.php?x={page_number * 30}&hair_color={hair_color_tag}"

    # begin setup
    software_names = [SoftwareName.FIREFOX.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

    user_agent = user_agent_rotator.get_random_user_agent()

    co = Options()
    co.add_argument("--headless")
    co.add_argument("--disable-gpu")
    co.add_argument("--window-size=1420, 1080")
    co.add_argument(f'user-agent={user_agent}')

    browser = webdriver.Firefox(
        options=co
    )

    # open up specific page number with specific hair color
    browser.get(page_url)
    page_html = browser.page_source
    soup = BeautifulSoup(page_html, features='html.parser')

    character_id_lines = soup.find_all(target="_blank")

    image_links = []

    # go to the characters on the page's specific page
    for character_id_line in character_id_lines:
        character_id_line = str(character_id_line)
        character_id = character_id_line[27:character_id_line.find("\" target")]
        character_url = f'https://www.animecharactersdatabase.com/characters.php?id={character_id}'

        browser.get(character_url)
        page_html = browser.page_source
        soup = BeautifulSoup(page_html, features='html.parser')

        og_image_line = str(soup.find(property='og:image'))
        og_image_link = og_image_line[15:og_image_line.find("\" property")]
        image_links.append(og_image_link)

        line = str(soup.find_all(id="tile1"))

        image_ids = re.findall('imgid=(.*?)\">', line)

        # get all the extra images
        for image_id in image_ids:
            image_link_page = f'https://www.animecharactersdatabase.com/photo.php?type_id=1&imgid={image_id}'

            browser.get(image_link_page)
            page_html = browser.page_source
            soup = BeautifulSoup(page_html, features='html.parser')
            browser.quit()

            image_link_line = str(soup.find(target="_blank"))
            image_link = image_link_line[9:image_link_line.find("\" target")]

            image_links.append(image_link)

    return image_links


def download_images_from_these_tags(start_page, end_page, hair_color, hair_color_tag, img_dir):
    """
    downloads images from https://www.animecharactersdatabase.com/ that have the given hair color
    :param start_page: page of search to start collecting images
    :param end_page: page of search to stop collecting images
    :param hair_color: hair color
    :param hair_color_tag: an integer that is tag for hair color for the website.
    :param img_dir: directory to save images to

    for example, if start_page_num=0, end_page_num=1, hair_color_tag=5,
    it would download the images from the first page of the search results given the tag being green hair, and
    save the images to images/class_
    """
    page_nums = [i for i in range(start_page, end_page)]
    length = len(page_nums)

    # collect image links from https://www.animecharactersdatabase.com/ that satisfy the tags
    with concurrent.futures.ProcessPoolExecutor() as executor:
        hair_color_tags = [hair_color_tag] * length

        total_image_links = list(
            tqdm(
                executor.map(get_image_links, page_nums, hair_color_tags),
                total=length,
                file=sys.stdout,
                desc=f"Image links collected (Hair color tag: {hair_color_tag})"
            )
        )

    # flatten the list
    total_image_links = [item for sublist in total_image_links for item in sublist]

    # save the images in the image_directory/class directory
    with concurrent.futures.ThreadPoolExecutor() as executor:
        length = len(total_image_links)
        img_names = [i for i in range(length)]

        list(
            tqdm(
                executor.map(
                    download_image_from, total_image_links, [f'{img_dir}/{hair_color}'] * length, img_names
                ),
                total=length,
                file=sys.stdout,
                desc=f"Images saved (Hair color tag: {hair_color_tag})"
            )
        )


def main(start_page, end_page, img_dir='images'):
    hair_colors = {'black': 1, 'blonde': 2, 'blue': 3, 'brown': 4, 'green': 5, 'gray': 6, 'orange': 7, 'purple': 8,
                   'red': 9, 'white': 10, 'pink': 12}

    for hair_color, hair_color_tag in hair_colors.items():
        os.mkdir(f'./{img_dir}/{hair_color}')

        download_images_from_these_tags(
            start_page=start_page,
            end_page=end_page,
            hair_color=hair_color,
            hair_color_tag=hair_color_tag,
            img_dir=img_dir
        )


if __name__ == '__main__':
    START_PAGE = 0
    END_PAGE = 30
    IMG_DIR = 'images'

    os.mkdir(f'./{IMG_DIR}')

    main(START_PAGE, END_PAGE, img_dir=IMG_DIR)
