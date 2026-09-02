from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv

from signature_identification import SignatureIdentificationService


class SignatureFlow(StatesGroup):
    waiting_signature = State()
    waiting_full_name = State()


CALLBACK_CHECK = "menu:check"
CALLBACK_HELP = "menu:help"
CALLBACK_STATS = "menu:stats"
CALLBACK_CANCEL = "flow:cancel"
CALLBACK_HOME = "menu:home"

router = Router(name="signature-bot")


def main_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔎 Проверить подпись", callback_data=CALLBACK_CHECK)],
            [
                InlineKeyboardButton(text="📊 Статистика", callback_data=CALLBACK_STATS),
                InlineKeyboardButton(text="ℹ️ Помощь", callback_data=CALLBACK_HELP),
            ],
        ]
    )


def cancel_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data=CALLBACK_CANCEL)],
        ]
    )


def back_to_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ В меню", callback_data=CALLBACK_HOME)],
        ]
    )


def _is_image_document(message: Message) -> bool:
    document = message.document
    return bool(document and document.mime_type and document.mime_type.startswith("image/"))


async def _extract_image_bytes(message: Message) -> bytes:
    target: Any | None = None
    if message.photo:
        target = message.photo[-1]
    elif _is_image_document(message):
        target = message.document
    if target is None:
        raise ValueError("Message does not contain image content.")

    image_buffer = io.BytesIO()
    await message.bot.download(target, destination=image_buffer)
    return image_buffer.getvalue()


def _format_match_message(match: dict[str, Any]) -> str:
    return (
        "<b>Подпись найдена в базе</b>\n"
        f"ФИО: <b>{match['full_name']}</b>"
    )


def _format_new_signature_prompt() -> str:
    return (
        "<b>Подпись не распознана как существующая</b>\n"
        "Введите ФИО для регистрации нового профиля."
    )


async def _send_welcome(message: Message, state: FSMContext) -> None:
    await state.clear()
    text = (
        "<b>Бот идентификации подписей</b>\n"
        "1. Нажмите «Проверить подпись» или просто отправьте фото.\n"
        "2. Если подпись известна, я верну ФИО.\n"
        "3. Если подпись новая, запрошу ФИО и сохраню профиль."
    )
    await message.answer(text, reply_markup=main_inline_keyboard())


async def _send_help(chat_message: Message) -> None:
    text = (
        "<b>Как пользоваться</b>\n"
        "1. Отправьте фото подписи (как фото или как документ).\n"
        "2. Если подпись уже есть в базе, бот вернет ФИО.\n"
        "3. Если подпись новая, бот попросит ФИО и добавит новый профиль.\n\n"
        "Команды:\n"
        "/start - главное меню\n"
        "/check - режим ожидания подписи\n"
        "/stats - количество профилей в базе\n"
        "/cancel - отменить текущее действие"
    )
    await chat_message.answer(text, reply_markup=back_to_menu_keyboard())


async def process_signature_message(
    message: Message,
    state: FSMContext,
    signature_service: SignatureIdentificationService,
) -> None:
    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    progress_message = await message.answer("Обрабатываю подпись, подождите 1-3 секунды...")

    try:
        image_bytes = await _extract_image_bytes(message)
        query_vector = await asyncio.to_thread(signature_service.get_signature_vector, image_bytes)
        best_match = await asyncio.to_thread(signature_service.find_best_match, query_vector)
    except Exception as err:
        await progress_message.edit_text(
            "Не удалось обработать изображение. Проверьте, что отправлено четкое фото подписи.\n"
            f"Техническая ошибка: <code>{err}</code>",
            reply_markup=main_inline_keyboard(),
        )
        return

    threshold = float(signature_service.confidence_threshold)
    if best_match and float(best_match["confidence_score"]) > threshold:
        await state.clear()
        await progress_message.edit_text(
            _format_match_message(best_match),
            reply_markup=main_inline_keyboard(),
        )
        return

    source_ref = f"telegram://chat/{message.chat.id}/message/{message.message_id}"
    await state.set_state(SignatureFlow.waiting_full_name)
    await state.update_data(
        pending_vector=query_vector,
        pending_source_ref=source_ref,
    )
    await progress_message.edit_text(
        _format_new_signature_prompt(),
        reply_markup=cancel_inline_keyboard(),
    )


@router.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await _send_welcome(message, state)


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    await _send_help(message)


@router.message(Command("stats"))
async def stats_handler(
    message: Message,
    signature_service: SignatureIdentificationService,
) -> None:
    count = await asyncio.to_thread(signature_service.get_profiles_count)
    await message.answer(
        f"<b>Профилей в базе:</b> {count}",
        reply_markup=back_to_menu_keyboard(),
    )


@router.message(Command("check"))
async def check_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(SignatureFlow.waiting_signature)
    await message.answer(
        "Отправьте фото вашей подписи для идентификации",
        reply_markup=cancel_inline_keyboard(),
    )


@router.message(Command("cancel"))
async def cancel_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Действие отменено.", reply_markup=main_inline_keyboard())


@router.callback_query(F.data == CALLBACK_HOME)
async def home_callback(query: CallbackQuery, state: FSMContext) -> None:
    await query.answer()
    if query.message is not None:
        await _send_welcome(query.message, state)


@router.callback_query(F.data == CALLBACK_HELP)
async def help_callback(query: CallbackQuery) -> None:
    await query.answer()
    if query.message is not None:
        await _send_help(query.message)


@router.callback_query(F.data == CALLBACK_STATS)
async def stats_callback(
    query: CallbackQuery,
    signature_service: SignatureIdentificationService,
) -> None:
    await query.answer()
    if query.message is None:
        return
    count = await asyncio.to_thread(signature_service.get_profiles_count)
    await query.message.answer(
        f"<b>Профилей в базе:</b> {count}",
        reply_markup=back_to_menu_keyboard(),
    )


@router.callback_query(F.data == CALLBACK_CHECK)
async def check_callback(query: CallbackQuery, state: FSMContext) -> None:
    await query.answer()
    await state.set_state(SignatureFlow.waiting_signature)
    if query.message is not None:
        await query.message.answer(
            "Отправьте фото вашей подписи для идентификации",
            reply_markup=cancel_inline_keyboard(),
        )


@router.callback_query(F.data == CALLBACK_CANCEL)
async def cancel_callback(query: CallbackQuery, state: FSMContext) -> None:
    await query.answer("Отменено")
    await state.clear()
    if query.message is not None:
        await query.message.answer("Действие отменено.", reply_markup=main_inline_keyboard())


@router.message(SignatureFlow.waiting_full_name, F.photo)
@router.message(SignatureFlow.waiting_full_name, F.document)
async def waiting_name_got_image_handler(message: Message) -> None:
    await message.answer(
        "Сначала введите ФИО для текущей новой подписи или нажмите «Отмена».",
        reply_markup=cancel_inline_keyboard(),
    )


@router.message(SignatureFlow.waiting_full_name, F.text)
async def receive_full_name_handler(
    message: Message,
    state: FSMContext,
    signature_service: SignatureIdentificationService,
) -> None:
    full_name = (message.text or "").strip()
    if len(full_name) < 5:
        await message.answer("Введите корректное ФИО (минимум 5 символов).")
        return

    data = await state.get_data()
    pending_vector = data.get("pending_vector")
    source_ref = data.get("pending_source_ref", f"telegram://chat/{message.chat.id}")

    if not isinstance(pending_vector, list) or len(pending_vector) != 128:
        await state.clear()
        await message.answer(
            "Не нашел в памяти вектор подписи. Отправьте фото заново.",
            reply_markup=main_inline_keyboard(),
        )
        return

    profile_id = await asyncio.to_thread(
        signature_service.save_profile,
        full_name,
        pending_vector,
        source_ref,
    )
    await state.clear()
    await message.answer(
        "<b>Новый профиль сохранен</b>\n"
        f"ID: <code>{profile_id}</code>\n"
        f"ФИО: <b>{full_name}</b>",
        reply_markup=main_inline_keyboard(),
    )


@router.message(SignatureFlow.waiting_signature, F.photo)
@router.message(SignatureFlow.waiting_signature, F.document)
async def waiting_signature_handler(
    message: Message,
    state: FSMContext,
    signature_service: SignatureIdentificationService,
) -> None:
    if message.document and not _is_image_document(message):
        await message.answer(
            "Документ должен быть изображением. Отправьте фото подписи.",
            reply_markup=cancel_inline_keyboard(),
        )
        return
    await process_signature_message(message, state, signature_service)


@router.message(F.photo)
@router.message(F.document)
async def quick_image_handler(
    message: Message,
    state: FSMContext,
    signature_service: SignatureIdentificationService,
) -> None:
    if message.document and not _is_image_document(message):
        await message.answer("Нужен файл-изображение с подписью.", reply_markup=main_inline_keyboard())
        return
    await process_signature_message(message, state, signature_service)


@router.message()
async def fallback_handler(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state == SignatureFlow.waiting_signature.state:
        await message.answer(
            "Отправьте фото подписи или нажмите «Отмена».",
            reply_markup=cancel_inline_keyboard(),
        )
        return
    if current_state == SignatureFlow.waiting_full_name.state:
        await message.answer(
            "Введите ФИО для сохранения нового профиля или нажмите «Отмена».",
            reply_markup=cancel_inline_keyboard(),
        )
        return
    await message.answer(
        "Отправьте фото подписи или воспользуйтесь кнопками ниже.",
        reply_markup=main_inline_keyboard(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Telegram bot for handwritten signature identification")
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(project_dir / ".env"),
        help="Path to .env file that contains BOT_TOKEN.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(project_dir / "siamese_signature_best.pth"),
        help="Path to model weights file.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(project_dir / "signatures.sqlite3"),
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=65.0,
        help="Treat as same person if confidence_score is greater than this value.",
    )
    return parser


async def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)
    token = (os.getenv("BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            f"BOT_TOKEN is missing. Add it to {args.env_file} in format BOT_TOKEN=your_token"
        )

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    signature_service = SignatureIdentificationService(
        weights_path=args.weights,
        db_path=args.db_path,
        confidence_threshold=args.confidence_threshold,
    )

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    dp["signature_service"] = signature_service

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        signature_service.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
