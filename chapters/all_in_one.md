# Build a Neural Search App

## 👋 Introduction

This tutorial guides you through building your own neural search app using the [Jina framework](https://github.com/jina-ai/jina/). Don't worry if you're new to machine learning or search. We'll spell it all out right here.

![](./chapters/images/jinabox-southpark.gif)

Our example program will be a simple neural search engine for text. It will take a user's typed input, and return a list of lines from South Park that match most closely.

⚠️ Need help? Check out the [troubleshooting section].

## 🗝️ Key Concepts

First of all, read up on [Jina 101](https://github.com/jina-ai/jina/tree/master/docs/chapters/101) so you have a clear understanding of how Jina works. We're going to refer to those concepts a lot. We assume you already have some knowledge of Python and machine learning.

## 🧪 Try it Out!

Before going through the trouble of downloading, configuring and testing your app, let's get an idea of the finished product:

### Deploy with Docker

Jina has a pre-built Docker image with indexed data. You can run it with:

```bash
docker run -p 45678:45678 jinaai/hub.app.distilbert-southpark
```
Note: You'll need to run the Docker image before trying the steps below

#### Query with Jinabox

[Jinabox](https://github.com/jina-ai/jinabox.js/) is a simple web-based front-end for neural search.

![](./images/jinabox-southpark.gif)

1. Go to [jinabox](https://jina.ai/jinabox.js) in your browser
2. Ensure you have the server endpoint set to `http://localhost:45678/api/search`
3. Type a phrase into the search bar and see which South Park lines come up

#### Query with `curl`

Alternatively, you can open your shell and check the results via the RESTful API. The matched results are stored in `topkResults`.

```bash
curl --request POST -d '{"top_k": 10, "mode": "search", "data": ["text:hey, dude"]}' -H 'Content-Type: application/json' 'http://0.0.0.0:45678/api/search'
```

<details>
  <summary>See console output</summary>

```json  
{
  "search": {
    "docs": [
      {
        "weight": 1.0,
        "length": 1,
        "topkResults": [
          {
            "matchDoc": {
              "docId": 48,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Cartman[SEP]Hey, hey, did you see my iPad, Token?\n"
            },
            "score": {
              "value": 0.29252166,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 9322,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Stan[SEP]Oh thanks, dude.\n"
            },
            "score": {
              "value": 0.29002887,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 4053,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Kyle[SEP]Here's our cell phone, dude.\n"
            },
            "score": {
              "value": 0.28318727,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 2134,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Kyle[SEP]Oh hey dude.\n"
            },
            "score": {
              "value": 0.28181127,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 5083,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Henrietta[SEP]Thanks you guys.\n"
            },
            "score": {
              "value": 0.27215105,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 2823,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Cartman[SEP]Kyle, I want you to check his buddy list.\n"
            },
            "score": {
              "value": 0.27158132,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 4291,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Kyle[SEP]What are you talking about, dude!\n"
            },
            "score": {
              "value": 0.2715585,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 3386,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Kyle[SEP]Wow, dude, check it out!\n"
            },
            "score": {
              "value": 0.27094495,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 4613,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Kyle[SEP]Oh no, dude!\n"
            },
            "score": {
              "value": 0.2704847,
              "opName": "MinRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 890,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Stan[SEP]Hey you guys!\n"
            },
            "score": {
              "value": 0.27007768,
              "opName": "MinRanker"
            }
          }
        ],
        "mimeType": "text/plain",
        "text": "text:hey, dude"
      }
    ],
    "topK": 10
  }
}
```

</details>

## 🐍 Install

### Prerequisites

You'll need:

* Basic knowledge of Python
* Python 3.7 or higher installed, and pip
* A Mac or Linux computer (we don't currently support Windows)
* 8 gigabytes or more of RAM
* Plenty of time - Indexing can take a while!

You should have also read the key concepts at the top of this page to get a good overview of how Jina and this example work.

### Clone the repo

Let's get the basic files we need to get moving:

```
git clone XXX/tutorial
cd tutorial
```

### Cookiecutter

```
pip install -U cookiecutter && cookiecutter gh:jina-ai/cookiecutter-jina
```

We use [cookiecutter](https://github.com/cookiecutter/cookiecutter) to spin up a basic Jina app and save you having to do a lot of typing and setup. 

For our South Park example, we recommend the following settings:

* Project name: `South Park`
* Project slug: `south_park`
* Task type: `nlp`
* Index type: `strings`

All other fields you can just fill in however you please.

## 📂 Files and Folders

After running `cookiecutter`, you should see a bunch of files in the `south_park` folder:

| File               | What it Does                                                             |
| ---                | ---                                                                      |
| `app.py`           | The main Python script where you initialize and pass data into your Flow |
| `Dockerfile`       | Lets you spin up a Docker instance running your app                      |
| `flows/`           | Folder to hold your Flows                                                |
| `pods/`            | Folder to hold your Pods                                                 |
| `README`           | The index to this tutorial                                               |
| `requirements.txt` | A list of requirements for `pip`                                         |

In the `flows/` folder we can see `index.yml` and `query.yml` - these define the indexing and querying Flows for your app.

In `pods/` we see `chunk.yml`, `craft.yml`, `doc.yml`, and `encode.yml` - these Pods are called from the Flows to process data for indexing or querying.

### Install Requirements

In your terminal:

```
cd south_park
pip install -r requirements.txt
```

⚠️ Now we're going to get our hands dirty, and if we're going to run into trouble, this is where we'll find it. If you hit any snags, check our **[troubleshooting](./troubleshooting.md)** section!

## Prepare the Data

Our goal is to find out who said what in South Park episodes when a user queries a phrase. The [SouthPark dataset](https://github.com/BobAdamsEE/SouthParkData/) contains the characters and lines from seasons 1 to 19. Many thanks to [BobAdamsEE](https://github.com/BobAdamsEE) for sharing this awesome resource!👏

Now let's ensure we're back in our base folder and download and process this dataset by running:

```bash
cd ..
bash ./get_data.sh
```

<details>
  <summary>See console output</summary>

```bash
Cloning into './south_park/data'...
remote: Enumerating objects: 3852, done.
remote: Total 3852 (delta 0), reused 0 (delta 0), pack-reused 3852
Receiving objects: 100% (3852/3852), 5.11 MiB | 2.37 MiB/s, done.
Resolving deltas: 100% (40/40), done.
```

</details>

## Load the Data

Now that `get_data.sh` has downloaded the data (and called `process_data.py` to process it), we've got `character-lines.csv`. We need to pass this file into `app.py`. `app.py` is a little too simple out of the box, so we'll have to make some changes:

### Check the Data

Let's just make sure the file has everything we want:

```shell
head data/character-lines.csv
```

You should see output like:

```csv
Stan! I don't wanna talk about it, I jus' wanna leave.
Mysterion! Mrs.
Sally! Pa-pa.
Canadians! We want respect!
Phillip! That smelly Saddam Hussein.
Cartman! Strike me down while you can!
Morpheus! What if I were to tell you.
Kanye! Yep, got it.
Jimbo! Here we are at Shafer's Crossing, lookin' for some animals.
Kyle! it's okay.
```

In the lines above, `!` acts as a separator between the character and what they say.

### Add `filepath`

In the `index` function, we currently have:

```python
    with f:
        f.index_lines(['abc', 'cde', 'efg'], batch_size=64, read_mode='rb', size=num_docs)
```

As you can see, this indexes just 3 strings. Let's load up our South Park file instead with the `filepath` parameter:

```python
    with f:
        f.index_lines(filepath='data/character-lines.csv', batch_size=64, read_mode='r', size=num_docs)
```

Note we've also changed `read_mode` to `r`, since we're reading strings, not bytes.

### Index Fewer Documents

While we're here, let's reduce the number of documents we're indexing, just to speed things up while we're testing. We don't want to spend hours indexing only to have issues later on!

In the section above the `config` function, let's change:

```python
num_docs = os.environ.get('MAX_DOCS', 50000)
```

to:

```python
num_docs = os.environ.get('MAX_DOCS', 500)
```

That should speed up our testing by a factor of 100! Once we've verified everything works we can set it back to `50000` to index more of our dataset. If it still seems too slow, reduce that number down to 50 or so.

## Run the Flows

Now that we've got the code to load our data, we're going to dive into writing our app and running our Flows!

### Index Mode

First up we need to build up an index of our file. We'll search through this index when we use the query Flow later.

```bash
python app.py index
```

<details>
<summary>See console output</summary>

```console
index [====                ] 📃    256 ⏱️ 52.1s 🐎 4.9/s      4      batch        encoder@273512[I]:received "control" from gateway▸crafter▸encoder-head▸encoder-2▸⚐
        encoder@273512[I]:received "index" from gateway▸crafter▸⚐               
        encoder@273516[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-2▸⚐
        encoder@273525[I]:received "index" from gateway▸crafter▸encoder-head▸⚐    
      chunk_idx@273529[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-2▸encoder-tail▸⚐
      chunk_idx@273537[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-2▸encoder-tail▸chunk_idx-head▸⚐
      chunk_idx@273529[I]:received "control" from gateway▸crafter▸encoder-head▸encoder-2▸encoder-tail▸chunk_idx-head▸chunk_idx-1▸⚐
      chunk_idx@273533[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-2▸encoder-tail▸chunk_idx-head▸chunk_idx-1▸⚐
       join_all@273549[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-2▸encoder-tail▸chunk_idx-head▸chunk_idx-1▸chunk_idx-tail▸⚐
       join_all@273549[I]:collected 2/2 parts of IndexRequest                    
index [=====               ] 📃    320 ⏱️ 71.2s 🐎 4.5/s      5      batch        encoder@273512[I]:received "control" from gateway▸crafter▸encoder-head▸encoder-1▸⚐
        encoder@273512[I]:received "index" from gateway▸crafter▸⚐
        encoder@273516[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸⚐
        encoder@273520[I]:received "index" from gateway▸crafter▸encoder-head▸⚐    
      chunk_idx@273529[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸⚐                        
      chunk_idx@273541[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸⚐
      chunk_idx@273529[I]:received "control" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸chunk_idx-2▸⚐
      chunk_idx@273533[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸chunk_idx-2▸⚐                           
       join_all@273549[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸chunk_idx-2▸chunk_idx-tail▸⚐
       join_all@273549[I]:collected 2/2 parts of IndexRequest                       
index [======              ] 📃    384 ⏱️ 71.4s 🐎 5.4/s      6      batch        encoder@273512[I]:received "control" from gateway▸crafter▸encoder-head▸encoder-1▸⚐
        encoder@273516[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸⚐
      chunk_idx@273529[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸⚐
      chunk_idx@273537[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸⚐
      chunk_idx@273529[I]:received "control" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸chunk_idx-1▸⚐
      chunk_idx@273533[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸chunk_idx-1▸⚐
       join_all@273549[I]:received "index" from gateway▸crafter▸encoder-head▸encoder-1▸encoder-tail▸chunk_idx-head▸chunk_idx-1▸chunk_idx-tail▸⚐
       join_all@273549[I]:collected 2/2 parts of IndexRequest
```

</details>

### Search Mode

Run:

```bash
python app.py search
```

After a while you should see the console stop scrolling and display output like:

```console
Flow@85144[S]:flow is started at 0.0.0.0:65481, you can now use client to send request!
```

⚠️  Be sure to note down the port number. We'll need it for `curl` and jinabox! In our case we'll assume it's `65481`, and we use that in the below examples. If your port number is different, be sure to use that instead.

ℹ️  `python app.py search` doesn't pop up a search interface - for that you'll need to connect via `curl`, Jinabox, or another client.

### Searching Data

Now that the app is running in search mode, we can search from the web browser with Jinabox or the terminal with `curl`:

#### Jinabox

![](./images/jinabox-southpark.gif)
 
1. Go to [jinabox](https://jina.ai/jinabox.js) in your browser
2. Ensure you have the server endpoint set to `http://localhost:65481/api/search`
3. Type a phrase into the search bar and see which South Park lines come up

#### Curl

`curl` will spit out a *lot* of information in JSON format - not just the lines you're searching for, but all sorts of metadata about the search and the lines it returns. Look for the lines starting with `"matchDoc"` to find the matches.

```bash
curl --request POST -d '{"top_k": 10, "mode": "search", "data": ["text:hey, dude"]}' -H 'Content-Type: application/json' 'http://0.0.0.0:65481/api/search'
```

<details>
<summary>See console output</summary>

```json
{
  "search": {
    "docs": [
      {
        "chunks": [
          {
            "chunkId": 2859771895,
            "embedding": {},
            "weight": 1.0,
            "length": 1,
            "topkResults": [
              {
                "matchChunk": {
                  "docId": 454,
                  "chunkId": 797264081,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    8,
                    55
                  ]
                },
                "score": {
                  "value": 2.6883826,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 287,
                  "chunkId": 1142950297,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    8,
                    69
                  ]
                },
                "score": {
                  "value": 2.747776,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 418,
                  "chunkId": 3396351234,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    8,
                    33
                  ]
                },
                "score": {
                  "value": 2.7522104,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 158,
                  "chunkId": 1398208945,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    8,
                    25
                  ]
                },
                "score": {
                  "value": 2.9864397,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 345,
                  "chunkId": 3441934356,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    13,
                    40
                  ]
                },
                "score": {
                  "value": 2.994561,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 42,
                  "chunkId": 2326393068,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    5,
                    30
                  ]
                },
                "score": {
                  "value": 3.03432,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 374,
                  "chunkId": 3848825176,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    5,
                    35
                  ]
                },
                "score": {
                  "value": 3.080478,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 169,
                  "chunkId": 174461633,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    6,
                    17
                  ]
                },
                "score": {
                  "value": 3.0987353,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 70,
                  "chunkId": 614007298,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    8,
                    38
                  ]
                },
                "score": {
                  "value": 3.1020787,
                  "opName": "NumpyIndexer"
                }
              },
              {
                "matchChunk": {
                  "docId": 102,
                  "chunkId": 3182665395,
                  "offset": 1,
                  "weight": 1.0,
                  "length": 2,
                  "mimeType": "text/plain",
                  "location": [
                    8,
                    21
                  ]
                },
                "score": {
                  "value": 3.1413307,
                  "opName": "NumpyIndexer"
                }
              }
            ],
            "mimeType": "text/plain",
            "location": [
              0,
              14
            ]
          }
        ],
        "weight": 1.0,
        "length": 1,
        "topkResults": [
          {
            "matchDoc": {
              "docId": 454,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Cartman! Wendy, don't forget: I'll tell my mom on you.\n"
            },
            "score": {
              "value": 0.74899185,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 287,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Michael! Yeah, so listen: call up Firkle and meet me at Village Inn.\n"
            },
            "score": {
              "value": 0.74896955,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 418,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Cartman! Hey-where are you guys?\n"
            },
            "score": {
              "value": 0.74896795,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 158,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Cartman! Oh, shit, dude!\n"
            },
            "score": {
              "value": 0.7488801,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 345,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "CityWokOwner! I'm not stereotype, okay!\n"
            },
            "score": {
              "value": 0.74887705,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 42,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Stan! No, dude, I feel worse!\n"
            },
            "score": {
              "value": 0.74886215,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 374,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Kyle! Hey, uh, Jimmy, can we talk?\n"
            },
            "score": {
              "value": 0.7488448,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 169,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Randy! Hey yeah.\n"
            },
            "score": {
              "value": 0.74883795,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 70,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Michael! Hey, you wanna play with me?\n"
            },
            "score": {
              "value": 0.7488367,
              "opName": "BiMatchRanker"
            }
          },
          {
            "matchDoc": {
              "docId": 102,
              "weight": 1.0,
              "mimeType": "text/plain",
              "text": "Cartman! Well hello.\n"
            },
            "score": {
              "value": 0.748822,
              "opName": "BiMatchRanker"
            }
          }
        ],
        "mimeType": "text/plain",
        "text": "text:hey, dude"
      }
    ],
    "topK": 10
  },
  "status": {}
}
```
</details>



## Troubleshooting

### Module not found error

Be sure to run `pip install -r requirements.txt` before beginning, and ensure you have lots of RAM/swap and space in your `tmp` partition (see below issues). This may take a while since there are a lot of prerequisites to install.

If this error still keeps popping up, look into the error logs to try to find which module it's talking about, and then run:

```sh
pip install <module_name>
```


### My Computer Hangs

Machine learning requires a lot of resources, and if your machine just hangs this is often due to running out of memory. To fix this, try [creating a swap file](https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-20-04/) if you use Linux. This isn't such an issue on macOS, since it allocates swap automatically.

### `ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device`

This is often due to your `/tmp` partition running out of space so you'll need to [increase its size](https://askubuntu.com/questions/199565/not-enough-space-on-tmp).

### `command not found`

For any of these errors you'll need to install the package onto your system. In Ubuntu this can be done with:

```sh
sudo apt-get install <package_name>
```

## 🎁 Wrap Up

In this tutorial you've learned:

* How to install Jina
* How to load data from files for indexing
* How to query data with `curl` and Jinabox
* The nitty-gritty behind Jina Flows and Pods

Now that you have a broad understanding of how things work, you can try out some of more [example tutorials](https://github.com/jina-ai/examples) to build image or video search, or stay tuned for our next set of tutorials that build upon your South Park app.
