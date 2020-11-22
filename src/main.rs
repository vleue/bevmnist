#![windows_subsystem = "windows"]
#![allow(clippy::type_complexity)]

use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    type_registry::TypeUuid,
    utils::BoxedFuture,
};
use tract_onnx::prelude::*;

#[derive(Debug, TypeUuid)]
#[uuid = "578fae90-a8de-41ab-a4dc-3aca66a31eed"]
pub struct OnnxModel {
    pub model: SimplePlan<
        TypedFact,
        Box<dyn TypedOp>,
        tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
    >,
}

#[derive(Default)]
pub struct OnnxModelLoader;

impl AssetLoader for OnnxModelLoader {
    fn load<'a>(
        &'a self,
        mut bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async move {
            let model = tract_onnx::onnx()
                .model_for_read(&mut bytes)
                .unwrap()
                .into_optimized()?
                .into_runnable()?;

            load_context.set_default_asset(LoadedAsset::new(OnnxModel { model }));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["onnx"]
    }
}

enum Event {
    Draw(i32, i32),
    Clear,
}

fn main() {
    let mut builder = App::build();
    builder
        .add_resource(WindowDescriptor {
            title: "bevmnist".to_string(),
            #[cfg(target_arch = "wasm32")]
            canvas: Some("#bevy-canvas".to_string()),
            ..Default::default()
        })
        .add_plugins(DefaultPlugins);

    #[cfg(target_arch = "wasm32")]
    builder.add_plugin(bevy_webgl2::WebGL2Plugin::default());

    builder
        .add_asset::<OnnxModel>()
        .init_asset_loader::<OnnxModelLoader>()
        .init_resource::<State>()
        .add_event::<Event>()
        .add_startup_system(setup)
        .add_system(drawing_mouse)
        .add_system(drawing_touch)
        .add_system(clear_action)
        .add_system(update_texture)
        .add_system(infer)
        .run();
}

const DRAWING_ZONE_SIZE: u32 = 600;
const INPUT_SIZE: u32 = 28;

#[derive(Default)]
struct State {
    model: Option<Handle<OnnxModel>>,
}

fn setup(
    commands: &mut Commands,
    mut state: ResMut<State>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
    asset_server: Res<AssetServer>,
) {
    state.model = Some(asset_server.load("model.onnx"));

    commands.spawn(UiCameraBundle::default());

    let color_none = materials.add(Color::NONE.into());

    let drawing_texture = textures.add(Texture::new_fill(
        Vec2::new(DRAWING_ZONE_SIZE as f32, DRAWING_ZONE_SIZE as f32),
        &[0, 0, 0, 255],
        bevy::render::texture::TextureFormat::Rgba8UnormSrgb,
    ));

    commands
        .spawn(NodeBundle {
            style: Style {
                margin: Rect::all(Val::Auto),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..Default::default()
            },
            material: color_none.clone(),
            ..Default::default()
        })
        .with(bevy::ui::FocusPolicy::Pass)
        .with_children(|parent| {
            parent
                .spawn(ImageBundle {
                    style: Style {
                        size: Size::new(
                            Val::Px(DRAWING_ZONE_SIZE as f32),
                            Val::Px(DRAWING_ZONE_SIZE as f32),
                        ),
                        ..Default::default()
                    },
                    material: materials.add(drawing_texture.into()),
                    ..Default::default()
                })
                .with_bundle((Interaction::None, bevy::ui::FocusPolicy::Block));
            parent
                .spawn(NodeBundle {
                    style: Style {
                        margin: Rect::all(Val::Auto),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        size: Size::new(
                            Val::Px(DRAWING_ZONE_SIZE as f32),
                            Val::Px(DRAWING_ZONE_SIZE as f32),
                        ),
                        ..Default::default()
                    },
                    material: color_none.clone(),
                    ..Default::default()
                })
                .with_children(|text_parent| {
                    text_parent.spawn(TextBundle {
                        text: Text {
                            value: "".to_string(),
                            font: asset_server.load("FiraMono-Medium.ttf"),
                            style: TextStyle {
                                font_size: DRAWING_ZONE_SIZE as f32,
                                color: Color::WHITE,
                                ..Default::default()
                            },
                        },
                        ..Default::default()
                    });
                });
        });
}

fn drawing_mouse(
    (mut reader, events): (Local<EventReader<CursorMoved>>, Res<Events<CursorMoved>>),
    mut texture_events: ResMut<Events<Event>>,
    drawable: Query<(&Interaction, &GlobalTransform, &Style)>,
) {
    for (interaction, transform, style) in drawable.iter() {
        if let Interaction::Clicked = interaction {
            let width = if let Val::Px(x) = style.size.width {
                x
            } else {
                0.
            };
            let height = if let Val::Px(x) = style.size.height {
                x
            } else {
                0.
            };
            for event in reader.iter(&events) {
                let x = event.position.x - transform.translation.x + width / 2.;
                let y = event.position.y - transform.translation.y + height / 2.;
                texture_events.send(Event::Draw(x as i32, y as i32));
            }
        }
    }
}

fn drawing_touch(
    (mut reader, events): (Local<EventReader<TouchInput>>, Res<Events<TouchInput>>),
    mut texture_events: ResMut<Events<Event>>,
    drawable: Query<(&GlobalTransform, &Style), With<Interaction>>,
) {
    for (transform, style) in drawable.iter() {
        let width = if let Val::Px(x) = style.size.width {
            x
        } else {
            0.
        };
        let height = if let Val::Px(x) = style.size.height {
            x
        } else {
            0.
        };
        for event in reader.iter(&events) {
            let x = event.position.x - transform.translation.x + width / 2.;
            let y = event.position.y - transform.translation.y + height / 2.;
            texture_events.send(Event::Draw(x as i32, y as i32));
        }
    }
}

fn clear_action(
    keyboard_input: Res<Input<KeyCode>>,
    mut texture_events: ResMut<Events<Event>>,
    mut display: Query<&mut Text>,
) {
    if keyboard_input.pressed(KeyCode::Space) {
        texture_events.send(Event::Clear);
        display.iter_mut().next().unwrap().value = "".to_string();
    }
}

fn update_texture(
    (mut reader, events): (Local<EventReader<Event>>, Res<Events<Event>>),
    materials: Res<Assets<ColorMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
    drawable: Query<
        &Handle<ColorMaterial>,
        (With<Style>, With<GlobalTransform>, With<Interaction>),
    >,
) {
    for event in reader.iter(&events) {
        let mat = drawable.iter().next().unwrap();
        let material = materials.get(mat).unwrap();
        let texture = textures
            .get_mut(material.texture.as_ref().unwrap())
            .unwrap();

        match event {
            Event::Draw(x, y) => {
                let pixel_size = DRAWING_ZONE_SIZE / INPUT_SIZE;
                for i in -(pixel_size as i32 / 2)..(pixel_size as i32 / 2 + 1) {
                    for j in -(pixel_size as i32 / 2)..(pixel_size as i32 / 2 + 1) {
                        set_pixel(x + i, (texture.size.y as i32 - y) + j, 255, texture);
                    }
                }
            }
            Event::Clear => {
                for x in 0..texture.size.x as i32 {
                    for y in 0..texture.size.y as i32 {
                        set_pixel(x, y, 0, texture);
                    }
                }
            }
        }
    }
}

fn set_pixel(x: i32, y: i32, color: u8, texture: &mut Texture) {
    if x as f32 > texture.size.x - 1. || x < 0 {
        return;
    }
    if y as f32 > texture.size.y - 1. || y < 0 {
        return;
    }
    texture.data[(x as usize + (y as f32 * texture.size.x) as usize) * 4] = color;
    texture.data[(x as usize + (y as f32 * texture.size.x) as usize) * 4 + 1] = color;
    texture.data[(x as usize + (y as f32 * texture.size.x) as usize) * 4 + 2] = color;
}

fn get_pixel(x: i32, y: i32, texture: &Texture) -> u8 {
    if x as f32 > texture.size.x - 1. || x < 0 {
        return 0;
    }
    if y as f32 > texture.size.y - 1. || y < 0 {
        return 0;
    }
    texture.data[(x as usize + (y as f32 * texture.size.x) as usize) * 4]
}

fn infer(
    state: Res<State>,
    materials: Res<Assets<ColorMaterial>>,
    textures: Res<Assets<Texture>>,
    models: Res<Assets<OnnxModel>>,
    drawable: Query<&Handle<ColorMaterial>, (With<Interaction>, With<GlobalTransform>)>,
    mut display: Query<&mut Text>,
) {
    for mat in drawable.iter() {
        let material = materials.get(mat).unwrap();
        let texture = textures.get(material.texture.as_ref().unwrap()).unwrap();

        let pixel_size = (DRAWING_ZONE_SIZE / INPUT_SIZE) as i32;

        let image = tract_ndarray::Array4::from_shape_fn(
            (1, 1, INPUT_SIZE as usize, INPUT_SIZE as usize),
            |(_, _, y, x)| {
                let mut val = 0;
                for i in 0..pixel_size as i32 {
                    for j in 0..pixel_size as i32 {
                        val += get_pixel(
                            x as i32 * pixel_size + i,
                            y as i32 * pixel_size + j,
                            texture,
                        ) as i32;
                    }
                }
                if val > pixel_size * pixel_size / 2 {
                    1. as f32
                } else {
                    0. as f32
                }
            },
        )
        .into();

        if let Some(model) = models.get(state.model.as_ref().unwrap()) {
            let result = model.model.run(tvec!(image)).unwrap();

            if let Some((value, score)) = result[0]
                .to_array_view::<f32>()
                .unwrap()
                .iter()
                .cloned()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                if score > 10. {
                    display.iter_mut().next().unwrap().value = format!("{:?}", value);
                } else {
                    display.iter_mut().next().unwrap().value = "".to_string();
                }
            }
        }
    }
}
