# This code is based on https://github.com/floydwch/kaggle-cli/tree/master/kaggle_cli
# that distributed under the following licence:
#
# The MIT License (MIT)
#
# Copyright (c) 2017 floydwch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import re
import sys
import pickle
import progressbar
from mechanicalsoup import Browser


def login(username, password):
    config_dir_path = os.path.join(
        os.path.expanduser('~'),
        '.kaggle-cli'
    )
    pickle_path = os.path.join(
        config_dir_path,
        'browser.pickle'
    )

    if os.path.isfile(pickle_path):
        try:
            with open(pickle_path, 'rb') as file:
                data = pickle.load(file)
                if data['username'] == username and \
                        data['password'] == password:
                    return data['browser']
        except:
            pass

    browser = Browser()
    login_url = 'https://www.kaggle.com/account/login'

    login_page = browser.get(login_url)

    token = re.search(
        'antiForgeryToken: \'(?P<token>.+)\'',
        str(login_page.soup)
    ).group(1)

    login_result_page = browser.post(
        login_url,
        data={
            'username': username,
            'password': password,
            '__RequestVerificationToken': token
        }
    )

    error_match = re.search(
        '"errors":\["(?P<error>.+)"\]',
        str(login_result_page.soup)
    )

    if error_match:
        print(error_match.group(1))
        return

    if not os.path.isdir(config_dir_path):
        os.mkdir(config_dir_path, 0o700)

    with open(pickle_path, 'wb') as f:
        pickle.dump(dict(
            username=username, password=password, browser=browser
        ), f)

    return browser


def download(competition, file_name, username, password, local_filename=None):
    browser = login(username, password)
    base = 'https://www.kaggle.com'
    data_url = '/'.join([base, 'c', competition, 'data'])

    data_page = browser.get(data_url)

    if data_page.status_code == 404:
        print('competition not found')
        return

    data = str(data_page.soup)
    links = re.findall(
        '"url":"(/c/{}/download/[^"]+)"'.format(competition), data
    )

    if not links:  # fallback for inclass competition
        links = map(
            lambda link: link.get('href'),
            data_page.soup.find(id='data-files').find_all('a')
        )

    if not links:
        print('not found')

    for link in links:
        url = base + link
        if file_name is None or url.endswith('/' + file_name):
            if download_file(browser, url, local_filename) is False:
                return


def download_file(browser, url, local_filename = None):
    print('downloading {}\n'.format(url))
    local_filename = url.split('/')[-1] if local_filename == None else local_filename
    headers = {}
    done = False
    file_size = 0
    content_length = int(
        browser.request('head', url).headers.get('Content-Length')
    )

    bar = progressbar.ProgressBar()
    widgets = [local_filename, ' ', progressbar.Percentage(), ' ',
               progressbar.Bar(marker='#'), ' ',
               progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]

    if os.path.isfile(local_filename):
        file_size = os.path.getsize(local_filename)
        if file_size < content_length:
            headers['Range'] = 'bytes={}-'.format(file_size)
        else:
            done = True

    finished_bytes = file_size

    if file_size == content_length:
        print('{} already downloaded !'.format(local_filename))
        return
    elif file_size > content_length:
        print('Something wrong here, Incorrect file !')
        return
    else:
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=content_length).start()
        bar.update(finished_bytes)

    if not done:
        stream = browser.get(url, stream=True, headers=headers)

        if not is_downloadable(stream):
            warning = (
                'Warning: '
                'download url for file {} resolves to an html document '
                'rather than a downloadable file. \n'
                'Is it possible you have not '
                'accepted the competition\'s rules on the kaggle website?'
                .format(local_filename)
            )
            print('\n\n{}\n'.format(warning))
            return False

        with open(local_filename, 'ab') as f:
            for chunk in stream.iter_content(chunk_size=1024*1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    finished_bytes += len(chunk)
                    bar.update(finished_bytes)

        bar.finish()
        print('')

    return True

def is_downloadable(response):
    '''
    Checks whether the response object is a html page
    or a likely downloadable file.
    Intended to detect error pages or prompts
    such as kaggle's competition rules acceptance prompt.

    Returns True if the response is a html page. False otherwise.
    '''

    content_type = response.headers.get('Content-Type', '')
    content_disp = response.headers.get('Content-Disposition', '')

    if 'text/html' in content_type and 'attachment' not in content_disp:
        # This response is a html file
        # which is not marked as an attachment,
        # so we likely hit a rules acceptance prompt
        return False
    return True


