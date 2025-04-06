from fastapi import APIRouter

from api import authentication, chat, translation, chatpdf

router = APIRouter()

router.include_router(authentication.router, tags=["authentication"], prefix="/users")
router.include_router(chat.router, tags=["chat"], prefix="/chat")
# router.include_router(translation.router, tags=["translation"], prefix="/translation")
router.include_router(chatpdf.router, tags=["chatpdf"], prefix="/chatpdf")