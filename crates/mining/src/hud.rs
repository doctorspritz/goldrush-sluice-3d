use crate::app::Game;
use bevy::prelude::*;
use mining_sim::{session::GOLD_TARGET_G, ControlMode};

#[derive(Component)]
pub enum HudText {
    Scene,
    Mode,
    Numbers,
    Objective,
    Context,
    Message,
    Clock,
    Help,
}
#[derive(Component)]
pub struct HelpPanel;
#[derive(Component)]
pub struct PausePanel;
#[derive(Component)]
pub struct GoldBar;

fn text(
    commands: &mut Commands,
    value: &str,
    size: f32,
    left: f32,
    top: f32,
    color: Color,
    role: Option<HudText>,
) -> Entity {
    let mut entity = commands.spawn((
        Text::new(value),
        TextFont {
            font_size: size,
            ..default()
        },
        TextColor(color),
        ZIndex(2),
        Node {
            position_type: PositionType::Absolute,
            left: px(left),
            top: px(top),
            ..default()
        },
    ));
    if let Some(role) = role {
        entity.insert(role);
    }
    entity.id()
}
pub fn setup(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    fonts
        .insert(
            bevy::asset::AssetId::default(),
            Font::try_from_bytes(include_bytes!("../assets/fonts/Barlow-Medium.ttf").to_vec())
                .expect("Bundled Barlow font"),
        )
        .expect("Default font asset");
    let cream = Color::srgb(0.94, 0.91, 0.82);
    let muted = Color::srgb(0.63, 0.71, 0.68);
    let amber = Color::srgb(0.96, 0.68, 0.29);
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(20),
            top: px(20),
            width: px(350),
            height: px(112),
            padding: UiRect::all(px(18)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.035, 0.065, 0.062, 0.91)),
        BorderRadius::all(px(3)),
    ));
    text(
        &mut commands,
        "THE SLUICE MOOSE",
        24.0,
        38.0,
        36.0,
        cream,
        None,
    );
    text(
        &mut commands,
        "FIRST BEND  /  YOUR FIRST CLAIM",
        11.0,
        39.0,
        70.0,
        amber,
        Some(HudText::Scene),
    );
    text(
        &mut commands,
        "ON FOOT  •  EXTERNAL VIEW",
        12.0,
        39.0,
        102.0,
        muted,
        Some(HudText::Mode),
    );
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(20),
                top: px(146),
                column_gap: px(6),
                ..default()
            },
            ZIndex(5),
        ))
        .with_children(|row| {
            for (label, action) in [
                ("Demo", HudButton::Demo),
                ("New shift", HudButton::NewShift),
                ("Help", HudButton::Help),
                ("Pause", HudButton::Pause),
            ] {
                row.spawn((
                    Button,
                    action,
                    Node {
                        padding: UiRect::axes(px(12), px(9)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.04, 0.10, 0.10, 0.95)),
                ))
                .with_child((
                    Text::new(label),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(cream),
                    ButtonLabel(action),
                ));
            }
        });
    // Right rail: readable work order and a gold target, separate from controls.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(20),
                top: px(20),
                width: px(288),
                height: px(326),
                ..default()
            },
            BackgroundColor(Color::srgba(0.035, 0.065, 0.062, 0.94)),
            BorderRadius::all(px(3)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("SHIFT 01  /  FIRST GOLD"),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(amber),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    top: px(20),
                    ..default()
                },
            ));
            parent.spawn((
                Text::new("0.00 g"),
                TextFont {
                    font_size: 37.0,
                    ..default()
                },
                TextColor(cream),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    top: px(50),
                    ..default()
                },
                HudText::Numbers,
            ));
            parent
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(20),
                        right: px(20),
                        top: px(106),
                        height: px(3),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.15, 0.21, 0.19)),
                ))
                .with_children(|bar| {
                    bar.spawn((
                        Node {
                            width: percent(0),
                            height: percent(100),
                            ..default()
                        },
                        BackgroundColor(amber),
                        GoldBar,
                    ));
                });
            parent.spawn((
                Text::new("PLACE YOUR SLUICE"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(cream),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    right: px(20),
                    top: px(127),
                    ..default()
                },
                HudText::Objective,
            ));
            parent.spawn((
                Text::new("Hopper  0.000 m³\nWater  OFF\nFeed  OFF"),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(muted),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    top: px(229),
                    ..default()
                },
                HudText::Clock,
            ));
        });
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(20),
                bottom: px(20),
                right: px(20),
                height: px(78),
                ..default()
            },
            BackgroundColor(Color::srgba(0.035, 0.065, 0.062, 0.94)),
            BorderRadius::all(px(3)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(
                    "WASD walk   •   Right-drag look   •   E enter excavator   •   B place sluice",
                ),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(cream),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    right: px(20),
                    top: px(14),
                    ..default()
                },
                HudText::Context,
            ));
            parent.spawn((
                Text::new(
                    "V camera   /   F1 help   /   F5 save   /   F9 load   /   Esc mouse / pause",
                ),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(muted),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    top: px(50),
                    ..default()
                },
            ));
        });
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(cream),
        TextShadow::default(),
        Node {
            position_type: PositionType::Absolute,
            left: px(24),
            bottom: px(112),
            max_width: percent(70),
            ..default()
        },
        HudText::Message,
    ));
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: percent(50),
            top: percent(50),
            width: px(8),
            height: px(8),
            margin: UiRect::left(px(-4)),
            border: UiRect::all(px(1)),
            ..default()
        },
        BorderColor::all(Color::srgba(0.96, 0.92, 0.79, 0.8)),
        BorderRadius::all(px(4)),
    ));
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(20),
                top: px(196),
                width: px(530),
                height: px(500),
                padding: UiRect::all(px(24)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.025, 0.045, 0.044, 0.98)),
            HelpPanel,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(cream),
                HudText::Help,
            ));
        });
    commands.spawn((Node{position_type:PositionType::Absolute,width:percent(100),height:percent(100),display:Display::None,align_items:AlignItems::Center,justify_content:JustifyContent::Center,..default()},BackgroundColor(Color::srgba(0.025,0.04,0.035,0.75)),PausePanel))
        .with_children(|parent|{parent.spawn((Text::new("SHIFT PAUSED\n\nEsc to return to the claim\nF5 save shift  •  F9 restore shift\nCtrl + Q to quit"),TextFont{font_size:24.0,..default()},TextColor(cream),TextLayout::new_with_justify(Justify::Center)));});
}

#[allow(
    clippy::type_complexity,
    reason = "Bevy ParamSet expresses overlapping component access"
)]
pub fn update(
    game: Res<Game>,
    mut text: Query<(&HudText, &mut Text)>,
    mut panels: ParamSet<(
        Query<&mut Node, With<HelpPanel>>,
        Query<&mut Node, With<PausePanel>>,
        Query<&mut Node, With<GoldBar>>,
    )>,
) {
    let s = &game.session;
    let foot = s.player.mode == ControlMode::OnFoot;
    for (role, mut text) in &mut text {
        text.0=match role {
            HudText::Scene=>if game.demo {"FLOWING SLUICE  /  LIVE PAYDIRT".into()} else {"FIRST BEND  /  YOUR FIRST CLAIM".into()},
            HudText::Mode=>format!("{}  /  {}",if foot{"ON FOOT"}else{"EXCAVATOR"},if s.player.third_person{"EXTERNAL VIEW"}else{"FIRST PERSON"}),
            HudText::Numbers=>format!("{:.2} g",s.sluice.concentrate.gold_g),
            HudText::Objective=>{
                if game.demo { "FLOWING DEMO\n\nPaydirt and water are flowing.\nP pump · O feed · arrows set rates.\nNew shift starts a fresh claim.".into() }
                else if s.sluice.concentrate.gold_g>=GOLD_TARGET_G {format!("FIRST GOLD RECOVERED\n\n{:.2} g from your claim.\nKeep digging, or save your shift.",s.sluice.concentrate.gold_g)}
                else if s.sluice.placement.is_none(){"01   BUILD YOUR SETUP\n\nWalk to the yellow hopper.\nB previews the sluice; click to place.".into()}
                else if s.site.excavated.volume()<0.01{"02   BREAK GROUND\n\nEnter the excavator. Lower the\nbucket into the bank and hold Space.".into()}
                else if s.delivered_m3<0.01{"03   FEED THE HOPPER\n\nLift the full bucket, swing left,\nand hold J over the hopper.".into()}
                else if s.sluice.processed_m3<0.01{"04   START WASHING\n\nWalk to the sluice. P starts water,\nO opens the feed gate.".into()}
                else{format!("05   LIFT THE MATS\n\nStop feed with O, let it clear,\nstop water with P, then C to collect.\nTarget: {GOLD_TARGET_G:.0} g")}
            },
            HudText::Clock=>format!("Hopper  {:.3} / 0.850 m³\nWater  {}  ·  {:.2} L/s\nFeed  {}  ·  {:.3} L/s\nShift  {:02}:{:02}",s.sluice.hopper.volume(),if s.sluice.pump_on{"ON"}else{"OFF"},s.sluice.flow_m3s*1000.0,
                if s.sluice.feed_on{"ON"}else{"OFF"},s.sluice.feed_m3s*1000.0,s.sim_time as u64/60,s.sim_time as u64%60),
            HudText::Context=>{
                if let Some(p)=game.placement {
                    format!("SLUICE  {:.1}°   ·   {}\nR / Q rotate   ·   [ / ] slope   ·   Click build   ·   Esc cancel",p.slope_deg,game.placement_error.as_deref().unwrap_or("Ready to connect"))
                }else if foot && s.near_plant() && s.sluice.placement.is_some() {
                    "Click to look · Esc frees mouse · P water · O feed · Up / Down water rate · Left / Right feed rate · C lift mats when stopped".into()
                }else if foot {
                    format!("WASD walk   ·   Click to capture mouse / shovel   ·   Esc release   ·   E / Tab cab   ·   B place   ·   Shovel: click   ·   Hand load: {:.1} L",s.player.hand.volume()*1000.0)
                }else{
                    format!("WASD tracks   ·   Q / E swing   ·   R / F boom   ·   T / G stick   ·   Y / H curl   ·   Space cut   ·   J dump   |   Bucket {:.0}%",s.machine.bucket.volume()/mining_sim::machine::BUCKET_CAPACITY*100.0)
                }
            },
            HudText::Message=>s.message.clone(),
            HudText::Help=>"OPERATING THE CLAIM\n\nON FOOT\nWASD walk · Shift run · Click captures mouse
Move mouse to look · Esc releases it
Right-drag also looks while the cursor is free\nE / Tab enter cab · Left-click shovel · J pour\nB sluice preview · R / Q rotate · [ / ] slope\n\nIN THE EXCAVATOR\nWASD tracks · Q / E swing the upper body\nR / F raise / lower boom · T / G extend / tuck stick\nY / H curl / open bucket · Space / left-click cut · J dump\nTab leave cab · I engine · V first / third person\n\nAT THE PLANT\nP pump · O feed · C lift mats when stopped\nUp / Down water · Left / Right feed\nShift + arrows: larger rate changes\n\nF5 save · F9 restore · Esc mouse / pause · F1 help".into(),
        };
    }
    for mut node in &mut panels.p0() {
        node.display = if game.help {
            Display::Flex
        } else {
            Display::None
        };
    }
    for mut node in &mut panels.p1() {
        node.display = if s.paused {
            Display::Flex
        } else {
            Display::None
        };
    }
    for mut node in &mut panels.p2() {
        node.width =
            percent((s.sluice.concentrate.gold_g / GOLD_TARGET_G * 100.0).clamp(0.0, 100.0) as f32);
    }
}

#[derive(Component, Clone, Copy)]
pub enum HudButton {
    Demo,
    NewShift,
    Help,
    Pause,
}
#[derive(Component)]
pub struct ButtonLabel(HudButton);
pub fn buttons(
    mut game: ResMut<Game>,
    mut buttons: Query<(&Interaction, &HudButton, &mut BackgroundColor), Changed<Interaction>>,
    mut labels: Query<(&ButtonLabel, &mut Text)>,
) {
    for (interaction, action, mut background) in &mut buttons {
        background.0 = if *interaction == Interaction::Hovered {
            Color::srgb(0.16, 0.28, 0.26)
        } else {
            Color::srgb(0.04, 0.10, 0.10)
        };
        if *interaction != Interaction::Pressed {
            continue;
        }
        match action {
            HudButton::Demo => game.restart(true),
            HudButton::NewShift => game.restart(false),
            HudButton::Help => {
                game.help = !game.help;
                game.mouse_captured = false;
                game.stop_controls();
            }
            HudButton::Pause => {
                let paused = !game.session.paused;
                game.session.paused = paused;
                game.mouse_captured = false;
                game.stop_controls();
                game.command(mining_physics::worker::Command::Pause(paused));
            }
        }
    }
    for (label, mut text) in &mut labels {
        if matches!(label.0, HudButton::Pause) {
            text.0 = if game.session.paused {
                "Resume"
            } else {
                "Pause"
            }
            .into();
        }
    }
}
