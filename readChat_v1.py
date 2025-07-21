import asyncio
import json
import logging
import random
import requests
from typing import Any
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play,_play_with_simpleaudio
from time import sleep
import yt_dlp
import re
import aiosqlite
import sqlite3
import time
import datetime

from langchain_core.prompts import (
	ChatPromptTemplate,
	MessagesPlaceholder,
	PromptTemplate,
	SystemMessagePromptTemplate,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
import json

import twitchio
from twitchio import authentication, eventsub
from twitchio.ext import commands

import threading

import os

# NOTE: Consider reading through the Conduit examples
# Store and retrieve these from a .env or similar, but for example showcase you can just full out the below:
from config import (
	CLIENT_ID,
	CLIENT_SECRET,
	BOT_ID,
	OWNER_ID,
	OPENAI_API_KEY,
	OPENAI_MODEL_NAME,
	TTS_PATH,
	DB_PATH,
	MUSIC_PATH,
	DL_PATH,
	WORKING_PATH,
	TTS_GENERATE_PATH
)

IF_NEED_REPLY = None
IF_NEED_AUTO_RADIO = None
LAST_MESSAGE_TIME = None
IS_PLAYING_RADIO = None
ACTIVE_THREADS = list()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.chdir(WORKING_PATH)

LOGGER: logging.Logger = logging.getLogger(__name__)



def radio_instruction(user_id, dialog):

	return "現在有 !rules , !Jarvis"

def radio_rules(user_id, dialog):

	return "請主人善用 驚嘆號 rules"

def radio_station(user_id, dialog):
	message = "好的 我這就來推薦你一首歌"
	insert_dialog(user_id, dialog, message)
	
	response = ""
	song_file = "NULL"

	recommendation = llm_make_recommend(user_id)
	if recommendation["summary"] != "NULL":
		response = "根據你過往的對話紀錄 我推薦你 " + recommendation["song_name"] + " 因為" + recommendation["reason"]
		insert_summary(user_id, recommendation["summary"], recommendation["song_name"], recommendation["reason"])
		song_file = get_song_path(recommendation["song_name"])
	else:
		response = "再跟我多聊幾句吧 目前若林會唱的歌沒有值得推薦給你的"


	return [response, song_file]

def insert_summary(user_id, summary, song_name, reason):
	query = (
		"INSERT INTO recommendation (id, song_name, reason, recommend_time) VALUES (?, ?, ?, ?)"
	)

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute(query, (user_id, song_name, reason, int(time.time()) ) )
		conn.commit()

	query = (
		'UPDATE user SET summary = ?, dialog_status = 0 WHERE id = ?'
	)

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute(query, (summary, user_id) )
		conn.commit()


	return

def update_chat_history(user_id):
	query = (
		'SELECT * FROM current_session WHERE id = ?'
	)

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute(query, (user_id,) )
		res = cursor.fetchall()

		dialogs = list()
		for ele in res:
			dialogs.append([ele[0], ele[1], ele[2], ele[3]])


		query = (
			"INSERT INTO history (id, input, response, exchange_time) VALUES (?, ?, ?, ?)"
		)
		for ele in dialogs:
			cursor.execute(query, (ele[0], ele[1], ele[2], ele[3]) )
		conn.commit()

		query = (
			'DELETE FROM current_session WHERE id = ?'
		)
		cursor.execute(query, (user_id,) )
		conn.commit()

		cursor.execute('UPDATE user SET dialog_status = ? WHERE id = ?', (0, user_id))
		conn.commit()

	return

def get_song_list():
	song_list = list() #name, discription

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()

		query = (
			"SELECT name, description FROM music"
		)

		cursor.execute(query)
		res = cursor.fetchall()
		
		for ele in res:
			song_list.append([ele[0], ele[1]])
	

	return song_list

def get_song_path(song_name):
	path_name = ""
	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()

		query = (
			"SELECT path_name FROM music WHERE name = '" + song_name + "'"
		)

		cursor.execute(query)
		res = cursor.fetchall()
		
		path_name = res[0][0]


	return path_name

def llm_make_recommend(user_id):
	system_prefix = """
	You are an emotionally intelligent assistant that helps summarize chat interactions between a user and a music bot, 
	and recommends ***Only one*** suitable song based on the conversation's tone, theme, and emotional cues.
	You have access to tools to get the description of each song.
	You must only recommends one song in the given list and response in Traditional Chinese.

	### Available song list:
	"""
	song_list = ""
	count = 1
	song_dict = dict()
	for ele in get_song_list():
		song_list += str(count) + ". " + ele[1] + "  "
		song_dict[count] = ele
		count += 1
	system_prefix += song_list
	system_prefix += """
	
	### instructions
	1. Summarize the conversation briefly (2–3 sentences), capturing the user’s personality, mood, or topics of interest.
	2. Recommend one suitable song based on that summary and explain why
	3. Return in the following JSON format:
		{{
			"summary": "...",
			"recommended_song": 1,
			"reason": "..."
		}}

	"""

	@tool
	def song_description(song_option_number: int) -> str:
		"""Using Option number in the ### Available song list as input, you can obtain description about that song."""
		description = "This song did not exist"
		if song_option_number <= len(song_dict.keys()):
			description = "song name : " + song_dict[song_option_number][0]

		return description


	prompt = ChatPromptTemplate.from_messages([
		("system", system_prefix),
		("user", "{input}"),
		MessagesPlaceholder(variable_name="agent_scratchpad")
	])

	llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0)
	tools = [song_description]

	agent = create_openai_functions_agent(
		llm=llm,
		prompt=prompt,
		tools=tools,
	)
	agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


	output = agent_executor.invoke(
		{"input": get_chat_history(user_id)},
	)
	res = json.loads(output["output"])

	result = dict()
	result["summary"] = "NULL"
	result["song_name"] = "NULL"
	result["reason"] = "NULL"

	if "summary" in res.keys():
		result["summary"] = res["summary"]
		result["song_name"] = song_dict[int(res["recommended_song"])][0]
		result["reason"] = res["reason"]
	else:
		LOGGER.info("沒有正確產生推薦歌曲 - %s", user_name)


	return result

def get_chat_history(user_id):
	previous_logs = {
		"user_input":[],
		"exchange_time":[],
		"chatbot_response":[]
	}
	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()

		query = (
			"SELECT input, response, exchange_time FROM current_session WHERE id = ? "
			"ORDER BY exchange_time DESC LIMIT 10;"
		)

		cursor.execute(query, (user_id,))
		res = cursor.fetchall()
		
		for ele in res:
			previous_logs["user_input"].append(ele[0])
			previous_logs["exchange_time"].append(ele[2])
			previous_logs["chatbot_response"].append(ele[1])


	dialog = list()
	for index in reversed(range(len(previous_logs["user_input"]))):
		if int(time.time())-previous_logs["exchange_time"][index] < 864000:
			dialog.append(HumanMessage(content=previous_logs["user_input"][index]))
			dialog.append(AIMessage(content=previous_logs["chatbot_response"][index]))


	return dialog

def llm_intent_recognize(message):
	system_prefix = """

	You are an intent recognition module for a conversational music bot on Twitch, belongs to "若林".
	Given a user's message from the chat, your job is to classify the message into one of the predefined intents.

	Please respond **only with the intent name** (e.g., "chat_music_recommend") that best matches the message.  
	If the input doesn’t clearly match any intent, choose the most likely one.

	Available intents:

	1. "chat_music_recommend"
	→ The user wants to have a casual conversation with the bot, and after around ten messages, receive a personalized song recommendation based on the chat content.

	2. "chat_chitchat"
	→ The user is chatting.

	3. "rules"
	→ The user is asking the rules to use this bot.

	4. "instructions"  
	→ The user wants to know how to use specific instructions.

	Return only the intent name.

	"""
	full_prompt = ChatPromptTemplate.from_messages(
		[
			system_prefix,
			("user", "{input}"),
		]
	)

	llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
	chain = full_prompt | llm

	result = chain.invoke(
		{"input": message},
	)

	return result.content

def llm_chitchat(user_id, current_message):

	system_prefix = """

	You're designed to chitchat with users. Your name is "Jarvis", belongs to "若林".
	Make the messages looks like you are talking to them.

	#Make sure to follow these rules while chatting:

	1.Reply in a casual, warm, and expressive tone. Avoid formality.
	2.If possible, consider the context of previous messages to provide more accurate and relevant responses.
	3.Respond in Traditional Chinese.
	4.Keep responses within 30 characters.
	
	"""
	# 4.If you're unsure of the answer, respond with "Noanswer" or "沒答案"

	full_prompt = ChatPromptTemplate.from_messages(
		[
			system_prefix,
			MessagesPlaceholder(variable_name="messages"),
			("user", "{input}")
		]
	)
	llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
	chain = full_prompt | llm

	## find previous dialogs

	dialog = get_chat_history(user_id)

	result = chain.invoke(
		{
			"messages":dialog,
			"input": current_message
		}
	)

	# if result.content == "Noanswer.": 
	# 	return False,"eng"
	# elif result.content == "沒答案。":
	# 	return False,"ch"

	return result.content

def llm_paraphrase(user_name, sentence):

	system_prefix = """
	You're designed to paraphrase messages as a twitch chatbot.
	Make the messages looks like you are talking to them. 

	#Make sure to follow these rules when rephrasing:

	1.Identify yourself as "Jarvis" only when needed.
	2.Reply in a casual, warm, and expressive tone. Avoid formality.
	3.Respond in Traditional Chinese.
	4.Keep responses within 30 characters.
	5.If you need to mention user, call their name.
	"""

	full_prompt = ChatPromptTemplate.from_messages(
		[
			system_prefix,
			("user", "把'{input}'換句話跟'{name}'說"),
		]
	)

	llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
	chain = full_prompt | llm

	result = chain.invoke({
		"input": sentence,
		"name" : user_name
		},
	)

	return result.content

def check_if_new_user(user_id, name):
	status = True

	d = datetime.date.today()
	dt = datetime.datetime.combine(d, datetime.datetime.min.time())
	timestamp = dt.timestamp()
	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT name FROM user WHERE id = ?', (user_id,))
		
		rows = cursor.fetchall()
		if not rows:	
			cursor.execute('''INSERT INTO user (name, id, first_chat_time, dialog_status, summary, latest_chat_time) 
				VALUES (?, ?, ?, ?, ?, ?)''', (name, user_id, int(time.time()), 0, "與該使用者是初次見面", int(timestamp) ))
			conn.commit()
		else:
			if name is not rows[0][0]:
				cursor.execute('UPDATE user SET name = ? WHERE id = ?', (name, user_id))
				conn.commit()
			cursor.execute('UPDATE user SET latest_chat_time = ? WHERE id = ?', (int(timestamp), user_id))
			conn.commit()
			status = False

	return status

def insert_dialog(user_id, user_input, response):
	query = (
		"INSERT INTO current_session (id, input, response, exchange_time) VALUES (?, ?, ?, ?)"
	)

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute(query, (user_id, user_input, response, int(time.time()) ) )
		conn.commit()

	return

def count_exchange(user_id):
	count = 0

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()

		query = (
			"SELECT dialog_status FROM user WHERE id = '"
			+ str(user_id) + "'"
		)

		cursor.execute(query)
		res = cursor.fetchall()
		
		count = res[0][0]

	return count

def add_exchange(user_id):
	count = 0

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()

		query = (
			"SELECT (dialog_status) FROM user WHERE id = '"
			+ str(user_id) + "'"
		)

		cursor.execute(query)
		res = cursor.fetchall()
		count = res[0][0]

		cursor.execute('UPDATE user SET dialog_status = ? WHERE id = ?', (count+1, user_id))
		conn.commit()

	return count

def chat_song_recommand(user_name, url):
	pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11})'
	match = re.search(pattern, url)

	ydl_opts = {
		'quiet': True,  # 不輸出下載進度條
		'skip_download': True  # 不實際下載影片，只取得資訊
	}

	title = ""
	with yt_dlp.YoutubeDL(ydl_opts) as ydl:
		info_dict = ydl.extract_info(match.group(1), download=False)
		title = info_dict.get("title", None)

	with sqlite3.connect(DB_PATH) as conn:
		cursor = conn.cursor()
		cursor.execute('INSERT INTO rvc (name, title) VALUES (?, ?)', (user_name, title))
		
		conn.commit()

	return title

def play_music(file_name):
	LOGGER.info("Playing : " + MUSIC_PATH + r"\\" + file_name + ".mp3")

	sound = AudioSegment.from_mp3(MUSIC_PATH + r"\\" + file_name + ".mp3")
	play_obj = _play_with_simpleaudio(sound)
	play_obj.wait_done()

	return

def download_youtube_as_mp3(url, output_path=DL_PATH):
	pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11})'
	match = re.search(pattern, url)

	ydl_opts = {
		'format': 'bestaudio/best',
		'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # 自動以影片標題命名
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'mp3',
			'preferredquality': '192',  # kbps
		}],
		'quiet': False,
	}

	with yt_dlp.YoutubeDL(ydl_opts) as ydl:
		ydl.download([match.group(1)])

	return

def is_youtube_url(text):
	pattern = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}'
	return re.search(pattern, text) is not None

def text_to_speech(message, file_name):
	text = message
	url = TTS_GENERATE_PATH
	params = {
		"id": 0,
		"format": "mp3",
		"length": 1.1,
		"text": text,
	}

	response = requests.get(url, params=params)
	if response.status_code == 200:
		with open(TTS_PATH + file_name, "wb") as f:
			f.write(response.content)
	else:
		LOGGER.warning("API calls failed:", response.status_code)
	# tts = gTTS(text=message, lang='zh')
	# tts.save(TTS_PATH + file_name)

	sound = AudioSegment.from_mp3(TTS_PATH + file_name)
	silence = AudioSegment.silent(duration=2000)

	play_obj = _play_with_simpleaudio(sound + silence)
	play_obj.wait_done()

	return 

def generate_next_auto_radio():
	song_list = get_song_list()
	song_index = random.randrange(len(song_list))
	song_name = song_list[song_index][0]
	song_description = song_list[song_index][1]
	path_name = get_song_path(song_name)

	return song_description, path_name

intent_to_function = {
	"chat_music_recommend" : radio_station,
	"chat_chitchat" : llm_chitchat,
	"rules" : radio_rules,
	"instructions" : radio_instruction
}

class Bot(commands.Bot):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(**kwargs)

		global IF_NEED_AUTO_RADIO
		IF_NEED_AUTO_RADIO = False

		global IF_NEED_REPLY
		IF_NEED_REPLY = False

		global LAST_MESSAGE_TIME
		LAST_MESSAGE_TIME = time.time()

		global IS_PLAYING_RADIO
		IS_PLAYING_RADIO = False

		self.active_task = None
		self.bg_task = asyncio.create_task(self.idle_check_loop())

	async def event_ready(self):
		await self.add_component(GeneralCommands())

		print(f"✅ Bot 登入為：{self.user.name}")

	async def idle_check_loop(self):
		global IF_NEED_AUTO_RADIO
		global LAST_MESSAGE_TIME
		global IS_PLAYING_RADIO
		global ACTIVE_THREADS

		while True:
			await asyncio.sleep(10)
			ACTIVE_THREADS = [t for t in ACTIVE_THREADS if t.is_alive()]

			if not IF_NEED_AUTO_RADIO and not IS_PLAYING_RADIO: # 被中斷過 and 目前沒有在播放東西
				if self.active_task:
					self.active_task.cancel()

				await asyncio.sleep(10)
				now = time.time()
				if (now - LAST_MESSAGE_TIME) > 60:
					IF_NEED_AUTO_RADIO = True
					LOGGER.info("超過 60 秒沒人說話，自動開話題")
					self.active_task = asyncio.create_task(self.auto_start_topic())

	async def auto_start_topic(self):
		global IF_NEED_AUTO_RADIO
		global IS_PLAYING_RADIO
		global IF_NEED_REPLY
		global ACTIVE_THREADS

		next_tts_sentence = None
		thread = None
		try:
			while True:
				if IF_NEED_AUTO_RADIO and not IS_PLAYING_RADIO and len(ACTIVE_THREADS) == 0: # 更新播放內容
					next_audio = generate_next_auto_radio()
					next_tts_sentence = next_audio[0]
					thread = threading.Thread(target=play_music, args=(next_audio[1],))

					IS_PLAYING_RADIO = True
					text_to_speech(llm_paraphrase("若林寶們", next_tts_sentence), r"\tts.mp3")
					thread.start()
					ACTIVE_THREADS.append(thread)

				if IF_NEED_REPLY: # 自動播放話題任務的時候有人打字
					if thread.is_alive():
						text_to_speech("等若林這首歌唱完我再回覆你喔", r"\tts.mp3")
						IF_NEED_REPLY = False

				if not thread.is_alive(): # 目前內容播放結束
					IS_PLAYING_RADIO = False

				await asyncio.sleep(10) # 放出資源 監控聊天室

		except asyncio.CancelledError:
			LOGGER.info("自動播放話題任務被中斷")
			text_to_speech(next_tts_sentence, r"\tts.mp3")
			await asyncio.sleep(30)

	async def setup_hook(self) -> None:
		# Add our General Commands Component...
		await self.add_component(GeneralCommands())

		with open(".tio.tokens.json", "rb") as fp:
			tokens = json.load(fp)

		for user_id in tokens:
			if user_id == BOT_ID:
				continue

			# Subscribe to chat for everyone we have a token...
			chat = eventsub.ChatMessageSubscription(broadcaster_user_id=user_id, user_id=BOT_ID)
			await self.subscribe_websocket(chat)

	async def event_ready(self) -> None:

		LOGGER.info("Logged in as: %s", self.user)

	async def event_oauth_authorized(self, payload: authentication.UserTokenPayload) -> None:
		# Stores tokens in .tio.tokens.json by default; can be overriden to use a DB for example
		# Adds the token to our Client to make requests and subscribe to EventSub...
		await self.add_token(payload.access_token, payload.refresh_token)

		if payload.user_id == BOT_ID:
			return

		# Subscribe to chat for new authorizations...
		chat = eventsub.ChatMessageSubscription(broadcaster_user_id=payload.user_id, user_id=BOT_ID)
		await self.subscribe_websocket(chat)

class GeneralCommands(commands.Component):
	@commands.command()
	async def hi(self, ctx: commands.Context) -> None:
		"""Command that replys to the invoker with Hi <name>!

		!hi
		"""
		user_name = ctx.author.display_name
		LOGGER.info(user_name + " 向你問好")
		text_to_speech(user_name + " 向你問好", r"\tts.mp3")

		await ctx.send("SUBtember 嗨~ " + user_name)

	@commands.command()
	async def rules(self, ctx: commands.Context) -> None:

		message = "嗨 " + ctx.author.display_name + " 跟我累積十句對話(我有開口才算) 我會推薦一首歌給你聽喔"
		message = message + " 目前累計 " + str(count_exchange(ctx.author.id)) + " 句"

		await ctx.send("SUBtember " + message)

	@commands.command()
	async def Jarvis(self, ctx: commands.Context, *, message: str="default") -> None:
		"""Command which repeats what the invoker sends.
		!say <message>
		"""
		user_name = ctx.author.display_name
		user_id = ctx.author.id

		global IS_PLAYING_RADIO

		if(check_if_new_user(user_id, user_name)):
			insert_dialog(user_id, "嗨", user_name + "你好")
		
		if message == "default":
			message = "嗨!" + user_name + "想聊點什麼嗎?"
			if not IS_PLAYING_RADIO:
				IS_PLAYING_RADIO = True
				text_to_speech(message, r"\tts.mp3")
			insert_dialog(user_id, "嗨", message)

			IS_PLAYING_RADIO = False
			await ctx.send("SUBtember " + message)

		elif is_youtube_url(message):
			song_url = message
			song_title = chat_song_recommand(user_name, song_url)
			message = user_name + " 推薦的 " + song_title + "已經收到囉! 下次開台讓若林唱"
			download_youtube_as_mp3(song_url)
			insert_dialog(user_id, "推薦這首歌給若林主播唱", message[:40])
			add_exchange(user_id)

			if not IS_PLAYING_RADIO:
				IS_PLAYING_RADIO = True
				text_to_speech(message, r"\tts.mp3")
			
			IS_PLAYING_RADIO = False
			await ctx.send("SUBtember " + message)

		else:
			need_to_recomend = False
			music_file_name = ""

			user_input = message
			intent = llm_intent_recognize(user_input)
			# LOGGER.info("User intent: %s", intent)
			tmp_message = intent_to_function[intent](user_name, user_input)
			add_exchange(user_id)

			if isinstance(tmp_message, list):
				message = llm_paraphrase(user_name, tmp_message[0])
				if tmp_message[1] != "NULL":
					need_to_recomend = True
					music_file_name = tmp_message[1]
			else:
				if count_exchange(user_id) > 9:
					tmp_message = radio_station(user_id, user_input)
					message = llm_paraphrase(user_name, tmp_message[0])
					if tmp_message[1] != "NULL":
						need_to_recomend = True
						music_file_name = tmp_message[1]
				else:
					message = llm_paraphrase(user_name, tmp_message)
					
			insert_dialog(user_id, user_input, message[:40])


			if not IS_PLAYING_RADIO:
				IS_PLAYING_RADIO = True
				text_to_speech(message, r"\tts.mp3")
				IS_PLAYING_RADIO = False

			if need_to_recomend:
				update_chat_history(user_id)
				text_to_speech(user_name + "你跟我的對話已經滿十句了 讓我找找該推薦你什麼歌 等我一下下")
				thread = threading.Thread(target=play_music, args=(music_file_name,))
				thread_wait = True
				while True:
					if not IS_PLAYING_RADIO:
						thread_wait = False

					global ACTIVE_THREADS
					if not thread_wait and len(ACTIVE_THREADS) == 0:
						IS_PLAYING_RADIO = True
						text_to_speech("接下來這首是推薦給" + user_name + "的歌曲" + message, r"\tts.mp3")
						thread.start()
						ACTIVE_THREADS.append(thread)

					if not thread.is_alive(): # 目前內容播放結束
						IS_PLAYING_RADIO = False
						break

					await asyncio.sleep(5) # 放出資源 監控聊天室

			await ctx.send("SUBtember " + message)
			
			

	@commands.Component.listener()
	async def event_message(self, message: twitchio.ChatMessage) -> None:
		global IF_NEED_REPLY
		global IF_NEED_AUTO_RADIO
		global LAST_MESSAGE_TIME
		global IS_PLAYING_RADIO

		# IF_NEED_AUTO_RADIO = False

		if message.text.startswith("!"): # 使用者使用指令
			IF_NEED_REPLY = True
			LAST_MESSAGE_TIME = time.time()
			IF_NEED_AUTO_RADIO = False
			# IS_PLAYING_RADIO = True
		elif message.text.startswith("SUBtember"): # Bot回覆訊息
			IF_NEED_REPLY = False
			LAST_MESSAGE_TIME = time.time()
			IF_NEED_AUTO_RADIO = False
			# IS_PLAYING_RADIO = False
		# else: # 使用者普通聊天
			# IF_NEED_REPLY = True

		# print(message.text)


def main() -> None:
	twitchio.utils.setup_logging(level=logging.INFO)

	async def runner() -> None:
		async with Bot(
			client_id=CLIENT_ID,
			client_secret=CLIENT_SECRET,
			bot_id=BOT_ID,
			owner_id=OWNER_ID,
			prefix="!",
		) as bot:
			await bot.start()

	try:
		asyncio.run(runner())
	except KeyboardInterrupt:
		LOGGER.warning("Shutting down due to KeyboardInterrupt")







if __name__ == "__main__":
	main()