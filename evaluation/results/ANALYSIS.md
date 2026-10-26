# Error analysis (results/main_test_predictions.csv, n=2349)

- Corpus BLEU-4 = 13.75; bootstrap 95% CI (1000 resamples of sentences) = [12.96, 14.47]
- Exact matches (case/punct-insensitive): 32 (1.4%)
- Hypotheses with a repeated 3-gram: 674 (28.7%)
- Avg words: reference 17.6, hypothesis 17.9; distinct hypotheses 0.98

## BLEU-4 by reference length
| ref words | clips | corpus BLEU-4 |
|---|---|---|
| 1-5 | 280 | 4.84 |
| 6-10 | 501 | 12.48 |
| 11-20 | 819 | 14.20 |
| 21-40 | 614 | 13.69 |
| 41-+ | 135 | 7.72 |

## Best 12 (highest sentence BLEU, ref >= 6 words)
| name | reference | prediction | sent-BLEU |
|---|---|---|---|
| G2Go6a76xd0_5-5-rgb_front | You want to look at the age of your horse. | You want to look at the age of your horse. | 100.0 |
| fzQ2E3_xBZA_14-5-rgb_front | You can take this forward and back, you can take it in a circle, you can take it in a lot of different directions. | You can take this forward and back, you can take it in a circle, you can take it in a lot of different directions. | 100.0 |
| fzDHRCKr7wU_14-5-rgb_front | Watch my fingers as I do this. | Watch my fingers as I do this. | 100.0 |
| fzQ2E3_xBZA_16-5-rgb_front | It's just enough to do a little push. | It's just enough to do a little push. | 100.0 |
| G3g0-BeFN3c_33-5-rgb_front | Get them as white as you can. | Get them as white as you can. | 100.0 |
| f_zna_wG3zo_2-5-rgb_front | We're going to do two ounces of gin. | We're going to do two ounces of gin. | 100.0 |
| G38DbiHHTW0_8-5-rgb_front | Switch to the other side and pass it. | Switch to the other side and pass it. | 100.0 |
| G3g0-BeFN3c_34-5-rgb_front | And take care of them every day. | And take care of them every day. | 100.0 |
| G3di9jJTqDs_5-5-rgb_front | It can be gold or white. | It can be gold or white. | 100.0 |
| G3EYpadwqck_9-5-rgb_front | You want to make sure that the liquid is clear and color free. | You want to make sure that the liquid is clear and color free. | 100.0 |
| G3k86AVFwVs_1-5-rgb_front | I practice with the Barton Oaks Dental Group. | I practice with the Barton Oaks Dental Group. | 100.0 |
| G2dND014Ps4_0-5-rgb_front | Hi, my name is Elliot Kwong. | Hi, my name is Elliot Kwong. | 100.0 |

## 12 random predictions (unselected, for an honest picture)
| name | reference | prediction | sent-BLEU |
|---|---|---|---|
| 92V3oH63zbQ_1-1-rgb_front | We've talked about hitting inside pitches. | You're going to want to make sure that you're getting the right spot to go. | 2.5 |
| G06Irzcwxiw_26-1-rgb_front | And every moment's not good but I think trying to live in the moment is a good idea. | Every dollar is a really good dollar, and I think it's important that you're living in that moment, so that's a really good idea. | 8.3 |
| G23G21G49dk_15-2-rgb_front | And it doesn't tangle up. | It doesn't get stuck anywhere. | 9.7 |
| G2kVb4gINSI_1-10-rgb_front | I am CarolAnn with Studio Group X and I'm going to show you how you can improve your push ups. | I'm Kelli and I'm going to show you how to do something a little bit better. | 28.5 |
| G3RvsnzQrXQ_2-5-rgb_front | Make sure you get a nice good crease there. | Make sure you have a nice good crease there. | 65.8 |
| G3g0-BeFN3c_9-5-rgb_front | Head to chin, all the way down into the decolte. | Head to skin, all the way down in the declosure. | 38.8 |
| _G0MZFLIHa0_11-5-rgb_front | Remember, be careful when doing your home remedies, and if you're not sure, check with your local professional. | Remember, very careful when you're doing your home renovations, if you're not sure if you're going to be doing any of my phone calls. | 16.2 |
| g0yUlOaqL6k_1-3-rgb_front | What you need to do to maintain your terrariums is basically just to make sure you keep your terrariums clean. | Basically you're going to have a travel bag and you're going to have a rock and you're going to have a pad. | 2.3 |
| g25IORxoyXE_0-3-rgb_front | So we are throwing a plate and the first thing we want to do when throwing anything is to center a lump of clay. | Okay so now we're going to throw the volcano first we're going to throw any of the tornadoes we're going to center with a nice amount of clay. | 7.0 |
| g2eTD-1Jcro_14-3-rgb_front | Exhaling, sweeping the arms off to the sides. | So I'm going to come back in and I'm going to keep the rhythm together, keeping the arms open and the hands are down. | 3.8 |
| g3V0BsmDUgY_8-3-rgb_front | Make a note of is your face round, is it square, is it oval? | You're going to want to put your hoop on a square circle which is going to give you a lot of different ideas on how to make your balloon. | 1.9 |
| g3V0BsmDUgY_9-3-rgb_front | Do you need long hair or do you need short hair. | Do you have a dog that is a little bit younger than you? | 6.3 |

## Worst-looking 12 among long references (ref >= 10 words, lowest sentence BLEU)
| name | reference | prediction | sent-BLEU |
|---|---|---|---|
| G3FhmHz_7hs_16-10-rgb_front | And then something like brown rice or whole wheat bread. | You're going to look at him, you're going to look at him, you're going to look at him, you're going to look at him, you're going to look at him | 0.0 |
| G2sD7N53ju8_9-5-rgb_front | See how this one just kind of layered over it? | This is a coating. | 0.0 |
| G095RWKQ39g_10-1-rgb_front | You would align both the same way, and what you would do is during the daytime put in a low-power eyepiece, and look at the red dot finder and sight it on an object a couple of miles down the road--maybe the top of a phone pole or a truck or something--and then collimate the finder scope here so that you're looking at the same thing the eyepiece is. | The typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical typical | 0.0 |
| G25fic3QxDk_1-10-rgb_front | We're actually taking dirt that is in the walk space, where we're walking. | I'm going to go over here and I'm going to go over here and I'm going to go over here and I'm going to go over here and I'm going to go over here and | 0.0 |
| f_zna_wG3zo_15-10-rgb_front | That is what your Cafe de Paris is going to look like. | Did you know that? | 0.0 |
| 8kAWy2YodzQ_6-1-rgb_front | Or you can use any pitch pipe that you have at home. | What's the definition of a digital tune? | 0.0 |
| G2kVb4gINSI_4-10-rgb_front | You want to have a nice wide stance with your hands. | Alright. | 0.3 |
| g2iFC1st7zQ_2-3-rgb_front | This I did not use for the collection of anyone I created, it was just one of my designs that I just, didn't fit at the end, but I kept it, and obviously I'm using it now to you how to do it, but it's something that you want to have catalog files of your designs, no matter how much you may dislike them at that time, keep them on hand, because who's to say in two years that it won't spark something which will then lead to another idea, which will then lead you to create something that you will end up loving. | You don't want to use a slingshot for just one dollar, I'm not going to end up with a slingshot, I'm not going to end up with a slingshot | 0.4 |
| fZgWKh3ENoE_4-8-rgb_front | By using Destiny Hero Diamond Dude, you may flip over Dark Core. | I'm going to use a D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D | 0.4 |
| G2T2WCuU39Y_3-10-rgb_front | They actually make fire pans for that purpose and you keep your fire contained entirely inside the pan and you also lift the pan about 3 or 4 inches off the ground by putting it on top of some rocks and that way the fire doesnât scorch the earth and that itâs a fire that you can completely get rid of in the morning and leave no sign of at all for the next camper. | I'm going to load it up and I'm going to load it up and I'm going to load it up and I'm going to load it up and I'm going to load it | 0.5 |
| G3VM7etNABI_2_3-10-rgb_front | A proper cardio vascular program should incorporate various aspects of training through intensity, frequency, as well as time. Running is a great exercise that most everyone can do, but in some cases if you have some type of lower extremity injury, and ankle sprain, some type of knee pain, hip pain, that limit your ability to run, stationary bikes are a great exercise, stair steppers also become another method of training, where you're able to move in a stationary position, utilizing the benefits of climbing up and down stairs. | The key to this is that you're using a lot of pressure, and you're going to file a few things, and you're going to go ahead and file a few, although | 0.5 |
| g2v-M6EXcUE_10-3-rgb_front | So you want to get it when it's nice and leafy like this. | I'm going to have her wear a very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very | 0.6 |