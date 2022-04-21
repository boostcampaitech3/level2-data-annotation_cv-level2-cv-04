import argparse
import json
import os
import tqdm

from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State

from PIL import ImageOps, Image, ImageDraw, ImageFont
from itertools import chain
import plotly.express as px


def draw_polygon(img: Image, pts, illegibility: bool,tags):
    """이미지에 폴리곤을 그린다. illegibility의 여부에 따라 라인 색상이 다르다."""
    img_draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("NanumSquareRoundB.ttf",size=20)
    img_draw.rectangle([(pts[0][0],pts[0][1]-20),(pts[0][0]+200,pts[0][1]-20+20)], fill='yellow')
    img_draw.text((pts[0][0],pts[0][1]-20),tags,(0,0,0),font,align='left')

    pts = list(chain(*pts)) + pts[0]  # flatten 후 첫번째 점을 마지막에 붙인다.
    # 폴리곤 선 너비 지정이 안되어 line으로 표시
    img_draw.line(pts, width=3, fill=(0, 255, 255) if not illegibility else (255, 0, 255))


def read_img(image_name: str, target_h: int = 1000) -> Image:
    """이미지 로드 후 텍스트 영역 폴리곤을 표시하여 반환한다."""
    # load image, annotation
    img = Image.open(os.path.join(arg.d_path,image_name))
    img = ImageOps.exif_transpose(img)  # 이미지 정보에 따라 이미지를 회전

    # resize
    h, w = img.height, img.width
    ratio = target_h/h
    target_w = int(ratio * w)
    img = img.resize((target_w, target_h))

    # draw polygon
    for val in a_dict['images'][image_name]['words'].values():
        poly = val['points']
        tag_lan = val['language']
        tag_lan = ','.join(tag_lan) if type(tag_lan)==list else ''
        tag_tran = val["transcription"]
        tags = tag_lan+':'+ str(tag_tran)
        poly_resize = [[v * ratio for v in pt] for pt in poly]
        illegibility = val['illegibility']
        draw_polygon(img, poly_resize, illegibility,tags)

    return img


def initial_set():
    global a_dict,callback_list,img_idx,max_idx,fig_list,img_list
    img_idx = 0
    with open(arg.a_path,'r',encoding='utf-8') as a_json:
        a_dict = json.load(a_json)
        img_list = list(a_dict["images"])

    if arg.instant_mode:
        pass
    else:
        fig_list = []
        for img_name in tqdm.tqdm(img_list):
            img = read_img(img_name,arg.img_height)
            fig = px.imshow(img)
            fig_list.append(fig)

    if len(a_dict["images"])%(arg.row*arg.column) != 0:
        max_idx = len(a_dict["images"])//(arg.row*arg.column)
    else: 
        max_idx = len(a_dict["images"])//(arg.row*arg.column)-1

    # 기본 layout list
    layout_list = [
        html.H3(f'{img_idx}/{max_idx}', id='img_idx'),
        html.Button('prev', id='btn-prev', n_clicks=0),
        html.Button('next', id='btn-next', n_clicks=0),
        dcc.Input(
            id='idx_input',
            type='number', 
            placeholder='input image index', 
            min=0, 
            max=max_idx
        ),
        html.Button('go', id='btn-go', n_clicks=0),
    ]

    # graph layout 추가
    callback_list = [Output('img_idx', 'children')]
    for r in range(arg.row):            
        graph_list = []
        grid_template_columns = ''
        for c in range(arg.column):
            frame_id = c+r*arg.column
            graph_list.append(html.Div([
                html.H3(str(frame_id),id='img_name_%d' %frame_id),
                dcc.Graph(id='graph_%d' %frame_id),
                ]))
            grid_template_columns += arg.img_width + ' '
            callback_list.append(Output('graph_%d' %frame_id, 'figure'))
            callback_list.append(Output('img_name_%d' %frame_id, 'children'))
        layout_list.append(html.Div(graph_list,style={'display':'grid', 'grid-template-columns':grid_template_columns}))

    callback_list += [
        Output('idx_input', 'value'),
        Input('btn-prev', 'n_clicks'),
        Input('btn-next', 'n_clicks'),
        Input('btn-go', 'n_clicks'),
        State('idx_input', 'value')]
    app.layout = html.Div(layout_list)


def callback_set():
    @app.callback(*callback_list)
    def update_img(btn_n, btn_p, btn_g, go_idx: int):
        global img_idx

        change_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'btn-next' in change_id:  # 다음
            img_idx = 0 if img_idx == max_idx else img_idx + 1
        elif 'btn-prev' in change_id:  # 이전
            img_idx = max_idx if img_idx == 0 else img_idx - 1
        elif 'btn-go' in change_id:  # 특정 인덱스로 이동
            img_idx = int(go_idx)
        else:  # 초기화
            img_idx = 0

        output_list = [f'{img_idx}/{max_idx}']
        for r in range(arg.row):
            for c in range(arg.column):
                im_idx = img_idx*arg.row*arg.column + c+r*arg.column
                if im_idx < len(img_list):
                    if arg.instant_mode:
                        img_name = img_list[im_idx]
                        img = read_img(img_name,arg.img_height)
                        fig = px.imshow(img)
                        output_list.append(fig)
                    else:
                        output_list.append(fig_list[im_idx])
                    output_list.append(img_list[im_idx])
                else:
                    output_list.append(px.imshow(Image.new('P',(1,1))))
                    output_list.append(None)
        output_list += [img_idx]
        return tuple(output_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ho','--host',default='0.0.0.0')
    parser.add_argument('-p','--port',default='4000')
    parser.add_argument('-d','--d_path',default='../dataset/images',help='dataset path (default = "../dataset/images")')
    parser.add_argument('-a','--a_path',default='../dataset/ufo/annotation.json',help='annotation path (default = "../dataset/ufo/annotation.json")')
    parser.add_argument('-r','--row',default=2,help='')
    parser.add_argument('-c','--column',default=2,help='')
    parser.add_argument('-iw','--img_width',default='800px')
    parser.add_argument('-ih','--img_height',default=1000)
    parser.add_argument('-im','--instant_mode',default=True)
    arg = parser.parse_args()

    app = Dash(__name__)
    initial_set()
    callback_set()

    app.run_server(host=arg.host,port=arg.port)