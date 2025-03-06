from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Annotated
from io import BytesIO
from PIL import Image
from core.model import compare_images

router = APIRouter()

# @router.get('/search_dashboard/{search}')
# async def search_dashboard(search: str, session: AsyncSession = Depends(get_async_session), user=Depends(auth.verify_auth)):
#     if 1 <= len(search) <= 30:
#         search = search.lower()
#         query = select(User).where(func.lower(User.name).startswith(search))
#         users_found = (await session.execute(query)).fetchall()
#         users_json = {}
#         for u in users_found:
#             u = u[0]
#             if u.id == user:
#                 continue
#             query = select(Message).where(or_(and_(
#                 Message.from_user == u.id, Message.chat_id == user),
#                 and_(Message.from_user == user, Message.chat_id == u.id))).order_by(-Message.id)
#             last_msg = (await session.execute(query)).fetchone()
#             if last_msg:
#                 last_msg = last_msg[0]
#             else:
#                 last_msg = None
#             users_json[u.id] = search_json(u, last_msg)
#         return users_json
#     else:
#         raise HTTPException(status_code=HTTP_400_BAD_REQUEST,
#                             detail='Wrong length of string')
        
@router.post("/compare_images/")
async def compare_images_api(
    first_image: Annotated[UploadFile, File(...)],
    second_image: Annotated[UploadFile, File(...)]
):
    try:
        # Convert uploaded files to PIL images
        img1 = Image.open(BytesIO(await first_image.read())).convert("RGB")
        img2 = Image.open(BytesIO(await second_image.read())).convert("RGB")

        # Compare images
        sim_scores = await compare_images(img1, img2)

        return {"similarity": int(sim_scores*100000)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")