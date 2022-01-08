#![windows_subsystem = "windows"]
#![allow(clippy::type_complexity)]

use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    prelude::*,
    reflect::TypeUuid,
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
    Draw(Vec2),
    Clear,
}

#[cfg(not(target_arch = "wasm32"))]
mod sizes {
    pub const WINDOW_WIDTH: f32 = 1280.;
    pub const WINDOW_HEIGHT: f32 = 720.;
    pub const DRAWING_ZONE_STYLE: f32 = 600.0;
    pub const CLEAR_BUTTON_HEIGHT: f32 = 60.0;
}
#[cfg(target_arch = "wasm32")]
mod sizes {
    pub const WINDOW_WIDTH: f32 = 640.;
    pub const WINDOW_HEIGHT: f32 = 360.;
    pub const DRAWING_ZONE_STYLE: f32 = 300.0;
    pub const CLEAR_BUTTON_HEIGHT: f32 = 30.0;
}

use sizes::*;

fn main() {
    let mut builder = App::new();
    builder
        .insert_resource(WindowDescriptor {
            title: "bevmnist".to_string(),
            #[cfg(target_arch = "wasm32")]
            canvas: Some("#bevy-canvas".to_string()),
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins);

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
        .add_system(button_system)
        .run();
}

const INPUT_SIZE: u32 = 28;

enum PredictionState {
    Wait,
    Predict,
}

struct State {
    model: Handle<OnnxModel>,
    prediction_state: PredictionState,
}

impl FromWorld for State {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.get_resource::<AssetServer>().unwrap();
        State {
            prediction_state: PredictionState::Wait,
            model: asset_server.load("model.onnx"),
        }
    }
}

#[derive(Component)]
struct Drawable;
#[derive(Component)]
struct Prediction;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn_bundle(UiCameraBundle::default());

    let drawing_texture = asset_server.load("base-image.png");

    commands
        .spawn_bundle(NodeBundle {
            style: Style {
                margin: Rect::all(Val::Auto),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                flex_direction: FlexDirection::ColumnReverse,
                ..Default::default()
            },
            color: UiColor(Color::NONE),
            ..Default::default()
        })
        .insert(bevy::ui::FocusPolicy::Pass)
        .with_children(|parent| {
            parent
                .spawn_bundle(NodeBundle {
                    style: Style {
                        margin: Rect::all(Val::Auto),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..Default::default()
                    },
                    color: UiColor(Color::NONE),
                    ..Default::default()
                })
                .with_children(|predict| {
                    predict
                        .spawn_bundle(ImageBundle {
                            style: Style {
                                size: Size::new(
                                    Val::Px(DRAWING_ZONE_STYLE),
                                    Val::Px(DRAWING_ZONE_STYLE),
                                ),
                                ..Default::default()
                            },
                            image: UiImage(drawing_texture),
                            ..Default::default()
                        })
                        .insert(Drawable)
                        .insert_bundle((Interaction::None, bevy::ui::FocusPolicy::Block));
                    predict
                        .spawn_bundle(NodeBundle {
                            style: Style {
                                margin: Rect::all(Val::Auto),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                size: Size::new(
                                    Val::Px(DRAWING_ZONE_STYLE),
                                    Val::Px(DRAWING_ZONE_STYLE),
                                ),
                                ..Default::default()
                            },
                            color: UiColor(Color::NONE),
                            ..Default::default()
                        })
                        .with_children(|text_parent| {
                            text_parent
                                .spawn_bundle(TextBundle {
                                    text: Text::with_section(
                                        "",
                                        TextStyle {
                                            font: asset_server.load("FiraMono-Medium.ttf"),
                                            font_size: DRAWING_ZONE_STYLE / 2.0,
                                            color: Color::WHITE,
                                        },
                                        TextAlignment {
                                            ..Default::default()
                                        },
                                    ),
                                    ..Default::default()
                                })
                                .insert(Prediction);
                        });
                });
            parent
                .spawn_bundle(ButtonBundle {
                    style: Style {
                        size: Size::new(Val::Px(150.0), Val::Px(CLEAR_BUTTON_HEIGHT)),
                        // center button
                        margin: Rect {
                            left: Val::Auto,
                            top: Val::Px(CLEAR_BUTTON_HEIGHT / 3.),
                            right: Val::Auto,
                            bottom: Val::Auto,
                        },
                        // horizontally center child text
                        justify_content: JustifyContent::Center,
                        // vertically center child text
                        align_items: AlignItems::Center,
                        ..Default::default()
                    },
                    color: UiColor(Color::rgb(0.15, 0.15, 0.15)),
                    ..Default::default()
                })
                .with_children(|parent| {
                    parent.spawn_bundle(TextBundle {
                        text: Text::with_section(
                            "Clear",
                            TextStyle {
                                font: asset_server.load("FiraMono-Medium.ttf"),
                                font_size: CLEAR_BUTTON_HEIGHT * 2. / 3.,
                                color: Color::rgb(0.9, 0.9, 0.9),
                            },
                            TextAlignment {
                                ..Default::default()
                            },
                        ),
                        ..Default::default()
                    });
                });
        });
}

fn drawing_mouse(
    mut reader: EventReader<CursorMoved>,
    mut last_mouse_position: Local<Option<Vec2>>,
    mut texture_events: EventWriter<Event>,
    state: Res<State>,
    drawable: Query<(&Interaction, &GlobalTransform, &Style), With<Drawable>>,
) {
    for (interaction, transform, style) in drawable.iter() {
        if let Interaction::Hovered = interaction {
            if let PredictionState::Wait = state.prediction_state {
                texture_events.send(Event::Clear);
            }
        }
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
            for event in reader.iter() {
                if let Some(last_mouse_position) = *last_mouse_position {
                    let steps =
                        (last_mouse_position.distance(event.position) as u32 / INPUT_SIZE + 1) * 3;
                    for i in 0..steps {
                        let lerped =
                            last_mouse_position.lerp(event.position, i as f32 / steps as f32);
                        let x = lerped.x - transform.translation.x + width / 2.;
                        let y = lerped.y - transform.translation.y + height / 2.;

                        texture_events.send(Event::Draw(Vec2::new(x, y)));
                    }
                } else {
                    let x = event.position.x - transform.translation.x + width / 2.;
                    let y = event.position.y - transform.translation.y + height / 2.;
                    texture_events.send(Event::Draw(Vec2::new(x, y)));
                }

                *last_mouse_position = Some(event.position);
            }
        } else {
            *last_mouse_position = None;
        }
    }
}

fn drawing_touch(
    mut reader: EventReader<TouchInput>,
    mut texture_events: EventWriter<Event>,
    state: Res<State>,
    drawable: Query<(&GlobalTransform, &Style), With<Drawable>>,
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
        for event in reader.iter() {
            if let PredictionState::Wait = state.prediction_state {
                texture_events.send(Event::Clear);
            }
            let x = event.position.x - transform.translation.x + width / 2.;
            let y = event.position.y - transform.translation.y + height / 2.;
            texture_events.send(Event::Draw(Vec2::new(x, y)));
        }
    }
}

fn clear_action(
    keyboard_input: Res<Input<KeyCode>>,
    mut texture_events: EventWriter<Event>,
    mut display: Query<&mut Text, With<Prediction>>,
) {
    if keyboard_input.pressed(KeyCode::Space) {
        texture_events.send(Event::Clear);
        display.iter_mut().next().unwrap().sections[0].value = "".to_string();
    }
}

fn update_texture(
    mut reader: EventReader<Event>,
    mut textures: ResMut<Assets<Image>>,
    mut state: ResMut<State>,
    mut drawable: Query<(&bevy::ui::Node, &UiImage), With<Drawable>>,
) {
    let (node, image) = drawable.iter_mut().next().unwrap();
    let texture = match textures.get_mut(&image.0) {
        Some(image) => image,
        None => return,
    };
    for event in reader.iter() {
        match event {
            Event::Draw(pos) => {
                let radius = (1.3 * node.size.x / INPUT_SIZE as f32 / 2.) as i32;
                let scale = (texture.texture_descriptor.size.width as f32 / node.size.x) as i32;
                for i in -radius..(radius + 1) {
                    for j in -radius..(radius + 1) {
                        let target_point = Vec2::new(pos.x + i as f32, pos.y + j as f32);
                        if pos.distance(target_point) < radius as f32 {
                            for i in 0..=scale {
                                for j in 0..=scale {
                                    set_pixel(
                                        (target_point.x as i32) * scale + i,
                                        ((node.size.y as f32 - target_point.y) as i32) * scale + j,
                                        255,
                                        texture,
                                    )
                                }
                            }
                        }
                    }
                }
                state.prediction_state = PredictionState::Predict;
            }
            Event::Clear => {
                for x in 0..texture.texture_descriptor.size.width as i32 {
                    for y in 0..texture.texture_descriptor.size.height as i32 {
                        set_pixel(x, y, 0, texture);
                    }
                }
                state.prediction_state = PredictionState::Wait;
            }
        }
    }
}

fn set_pixel(x: i32, y: i32, color: u8, texture: &mut Image) {
    if x > texture.texture_descriptor.size.width as i32 - 1 || x < 0 {
        return;
    }
    if y > texture.texture_descriptor.size.height as i32 - 1 || y < 0 {
        return;
    }
    texture.data[(x as usize + (y as u32 * texture.texture_descriptor.size.width) as usize) * 4] =
        color;
    texture.data
        [(x as usize + (y as u32 * texture.texture_descriptor.size.width) as usize) * 4 + 1] =
        color;
    texture.data
        [(x as usize + (y as u32 * texture.texture_descriptor.size.width) as usize) * 4 + 2] =
        color;
}

fn get_pixel(x: i32, y: i32, texture: &Image) -> u8 {
    if x > texture.texture_descriptor.size.width as i32 - 1 || x < 0 {
        return 0;
    }
    if y > texture.texture_descriptor.size.height as i32 - 1 || y < 0 {
        return 0;
    }
    texture.data[(x as usize + (y as u32 * texture.texture_descriptor.size.width) as usize) * 4]
}

fn infer(
    state: Res<State>,
    textures: Res<Assets<Image>>,
    models: Res<Assets<OnnxModel>>,
    drawable: Query<&UiImage, With<Drawable>>,
    mut display: Query<&mut Text>,
) {
    if let PredictionState::Predict = state.prediction_state {
        for ui_image in drawable.iter() {
            let texture = textures.get(&ui_image.0).unwrap();

            let pixel_size = (texture.texture_descriptor.size.width as u32 / INPUT_SIZE) as i32;

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
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                },
            )
            .into();

            if let Some(model) = models.get(state.model.as_weak::<OnnxModel>()) {
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
                        display.iter_mut().next().unwrap().sections[0].value =
                            format!("{:?}", value);
                    } else {
                        display.iter_mut().next().unwrap().sections[0].value = "".to_string();
                    }
                }
            }
        }
    }
}

fn button_system(
    mut texture_events: EventWriter<Event>,
    mut interaction_query: Query<
        (&Interaction, &mut UiColor),
        (Changed<Interaction>, With<Button>),
    >,
    mut display: Query<&mut Text, With<Prediction>>,
) {
    for (interaction, mut color) in interaction_query.iter_mut() {
        match *interaction {
            Interaction::Clicked => {
                color.0 = Color::rgb(0.35, 0.75, 0.35);
                texture_events.send(Event::Clear);
                display.iter_mut().next().unwrap().sections[0].value = "".to_string();
            }
            Interaction::Hovered => {
                color.0 = Color::rgb(0.25, 0.25, 0.25);
            }
            Interaction::None => {
                color.0 = Color::rgb(0.15, 0.15, 0.15);
            }
        }
    }
}
