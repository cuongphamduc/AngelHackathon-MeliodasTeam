package com.vit.trafficvolumemap.ui.feature.main;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.location.Location;
import android.os.Build;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v4.app.FragmentActivity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.fastaccess.datetimepicker.DatePickerFragmentDialog;
import com.fastaccess.datetimepicker.callback.DatePickerCallback;
import com.fastaccess.datetimepicker.callback.TimePickerCallback;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.CameraPosition;
import com.google.android.gms.maps.model.Circle;
import com.google.android.gms.maps.model.CircleOptions;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.GenericTypeIndicator;
import com.google.firebase.database.ValueEventListener;
import com.vit.trafficvolumemap.BuildConfig;
import com.vit.trafficvolumemap.R;
import com.vit.trafficvolumemap.data.model.Camera;
import com.vit.trafficvolumemap.logger.CrashReportingTree;
import com.vit.trafficvolumemap.ui.util.Constant;
import com.vit.trafficvolumemap.ui.util.DateTimePickerHelper;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import timber.log.Timber;

public class MapsActivity extends FragmentActivity
        implements OnMapReadyCallback,
        GoogleApiClient.ConnectionCallbacks,
        GoogleApiClient.OnConnectionFailedListener,
        DatePickerCallback, TimePickerCallback {


    // ---------------------------------------------------------------------------------------------
    // FIELDS
    // ---------------------------------------------------------------------------------------------
    private static final int MY_REQUEST_INT = 177;

    private static boolean sIsGuess = false;

    private FirebaseDatabase mDatabase;
    private DatabaseReference mRef;

    private GoogleMap mMap;

    private GoogleApiClient mGoogleApiClient;

    private LatLng mCurrentLocation;

    private LatLng mLatLngSearchPosition;

    private List<Marker> mMarkerList = new ArrayList<>();

    private List<Circle> mCircleList = new ArrayList<>();


    // ---------------------------------------------------------------------------------------------
    // BIND VIEWS
    // ---------------------------------------------------------------------------------------------
    @BindView(R.id.button_guess)
    FloatingActionButton buttonGuess;

    @BindView(R.id.text_guess_time)
    TextView textGuessTime;


    // ---------------------------------------------------------------------------------------------
    // OVERRIDE METHODS
    // ---------------------------------------------------------------------------------------------
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        try {
            super.onCreate(savedInstanceState);

            initTimber();

            mGoogleApiClient = new GoogleApiClient.Builder(this).addConnectionCallbacks(this)
                    .addOnConnectionFailedListener(this)
                    .addApi(LocationServices.API)
                    .build();

            initView();
            initFirebase();
            fetchDataFromFirebase();

        } catch (Exception e) {
            Timber.e(e);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        mGoogleApiClient.connect();
    }

    @Override
    protected void onStop() {
        super.onStop();
        mGoogleApiClient.disconnect();
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        try {
            mMap = googleMap;

            mMap.getUiSettings().setZoomControlsEnabled(true);
            mMap.getUiSettings().setCompassEnabled(true);
            mMap.setTrafficEnabled(true);
            mMap.setBuildingsEnabled(true);

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                    != PackageManager.PERMISSION_GRANTED
                    && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
                    != PackageManager.PERMISSION_GRANTED) {

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    requestPermissions(new String[]{Manifest.permission.ACCESS_COARSE_LOCATION,
                            Manifest.permission.ACCESS_FINE_LOCATION}, MY_REQUEST_INT);
                }
                return;
            } else {
                mMap.setMyLocationEnabled(true);
            }

            onClickMap();
            formatInfoWindow();

        } catch (Exception e) {
            Timber.e(e);
        }

    }

    @Override
    public void onConnected(@Nullable Bundle bundle) {
        try {
            if (ActivityCompat.checkSelfPermission(this,
                    android.Manifest.permission.ACCESS_FINE_LOCATION)
                    == PackageManager.PERMISSION_GRANTED
                    && ActivityCompat.checkSelfPermission(this,
                    Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                final Location lastLocation =
                        LocationServices.FusedLocationApi.getLastLocation(mGoogleApiClient);
                if (lastLocation == null) {
                    return;
                }
                mCurrentLocation = new LatLng(lastLocation.getLatitude(), lastLocation.getLongitude());
                if (mLatLngSearchPosition == null) {
                    showCameraToPosition(mCurrentLocation, Constant.LEVEL_ZOOM_DEFAULT);
                }
            }
        } catch (Exception e) {
            Timber.e(e);
        }
    }

    @Override
    public void onConnectionSuspended(int i) {

    }

    @Override
    public void onConnectionFailed(@NonNull ConnectionResult connectionResult) {

    }

    @OnClick(R.id.button_guess)
    void onClickGuess(View view) {
        DatePickerFragmentDialog.newInstance(true)
                .show(getSupportFragmentManager(), DateTimePickerHelper.DATE_PICKER_FRAGMENT_DIALOG);
    }

    @OnClick(R.id.text_guess_time)
    void onClickTextGuessTime(View view) {
        textGuessTime.setVisibility(View.INVISIBLE);
        mRef.child("guess").child("is-guess").setValue(false);
    }

    @Override
    public void onDateSet(long date) {

    }

    @Override
    public void onTimeSet(long timeOnly, long dateWithTime) {
        try {
            textGuessTime.setText(DateTimePickerHelper.getDateAndTime(dateWithTime));
            textGuessTime.setVisibility(View.VISIBLE);
            writeTimeGuessToFirebase(DateTimePickerHelper.getDateTime(dateWithTime));
        } catch (Exception e) {
            Timber.e(e);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // PRIVATE METHODS
    // ---------------------------------------------------------------------------------------------

    /**
     * setup timber
     */
    private void initTimber() {
        if (BuildConfig.DEBUG) {
            Timber.plant(new Timber.DebugTree());
        } else {
            Timber.plant(new CrashReportingTree());
        }
    }

    /**
     * setup view
     */
    private void initView() {
        try {
            setContentView(R.layout.activity_maps);
            ButterKnife.bind(this);

            SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                    .findFragmentById(R.id.map);
            mapFragment.getMapAsync(this);
        } catch (Exception e) {
            Timber.e(e);
        }
    }

    /**
     * setup firebase
     */
    private void initFirebase() {
        mDatabase = FirebaseDatabase.getInstance();
        mRef = mDatabase.getReference();
    }

    /**
     * fetch data from Firebase
     */
    private void fetchDataFromFirebase() {
        try {
            mRef.addValueEventListener(new ValueEventListener() {
                @Override
                public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                    sIsGuess = (boolean) dataSnapshot.child("guess").child("is-guess").getValue();

                    List<Camera> cameraList = getDataFromFirebase(dataSnapshot.child("camera"));
                    displayTraffic(cameraList);
                }

                @Override
                public void onCancelled(@NonNull DatabaseError databaseError) {

                }
            });

        } catch (Exception e) {
            Timber.e(e);
        }
    }

    /**
     * get list camera from firebase
     *
     * @param dataSnapshot data of firebase
     * @return list camera
     */
    private List<Camera> getDataFromFirebase(DataSnapshot dataSnapshot) {
        List<Camera> cameraList = new ArrayList<>();

        try {
            GenericTypeIndicator<List<Camera>> genericTypeIndicator
                    = new GenericTypeIndicator<List<Camera>>() {
            };
            cameraList = dataSnapshot.getValue(genericTypeIndicator);
        } catch (Exception e) {
            Timber.e(e);
        }

        return cameraList;
    }

    /**
     * display traffic by circle around camera (position)
     *
     * @param cameraList list camera
     */
    private void displayTraffic(List<Camera> cameraList) {
        mMap.clear();
        for (final Camera camera : cameraList) {
            final LatLng position = new LatLng(camera.getLat(), camera.getLng());
            int color;
            float feature = (sIsGuess) ? camera.getGuess() : camera.getArea();
            String info = (sIsGuess) ? camera.toStringGuess() : camera.toString();

            if (feature < Constant.VOLUME_SMALL) {
                color = R.color.circle_green;
            } else if (feature > Constant.VOLUME_SMALL && feature < Constant.VOLUME_MEDIUN) {
                color = R.color.circle_orange;
            } else if (feature > Constant.VOLUME_MEDIUN && feature < Constant.VOLUME_LARGE) {
                color = R.color.circle_red;
            } else {
                color = R.color.circle_brown;
            }
            showCircleToGoogleMap(position, Constant.RADIUS_CIRCLE_TRAFFIC, color, info);

        }
    }

    /**
     * invisible all marker when touch map
     * <p>
     * show marker when touch circle
     */
    private void onClickMap() {
        try {
            mMap.setOnMapClickListener(new GoogleMap.OnMapClickListener() {
                @Override
                public void onMapClick(LatLng latLng) {
                    for (Marker m : mMarkerList) {
                        m.setVisible(false);
                    }
                }
            });

            mMap.setOnCircleClickListener(new GoogleMap.OnCircleClickListener() {
                @Override
                public void onCircleClick(Circle circle) {
                    showMarkerToGoogleMap(circle.getCenter(), (String) circle.getTag());
                }
            });
        } catch (Exception e) {
            Timber.e(e);
        }

    }

    /**
     * show marker when user click a position on map
     *
     * @param position position's marker
     */
    public void showMarkerToGoogleMap(LatLng position, String title) {
        try {
            String[] lines = title.split("\n", 2);

            MarkerOptions markerOptions = new MarkerOptions().position(position);
            markerOptions.icon(BitmapDescriptorFactory.fromResource(R.drawable.ic_location_active));
            markerOptions.title(lines[0]);
            markerOptions.snippet(lines[1]);

            Marker marker = mMap.addMarker(markerOptions);
            marker.showInfoWindow();

            mMarkerList.add(marker);

        } catch (Exception e) {
            Timber.e(e);
        }
    }

    /**
     * format info display on markers
     */
    private void formatInfoWindow() {
        try {
            mMap.setInfoWindowAdapter(new GoogleMap.InfoWindowAdapter() {

                @Override
                public View getInfoWindow(Marker arg0) {
                    return null;
                }

                @Override
                public View getInfoContents(Marker marker) {

                    LinearLayout info = new LinearLayout(getApplicationContext());
                    info.setOrientation(LinearLayout.VERTICAL);

                    TextView title = new TextView(getApplicationContext());
                    title.setTextColor(Color.BLACK);
                    title.setGravity(Gravity.CENTER);
                    title.setTypeface(null, Typeface.BOLD);
                    title.setText(marker.getTitle());

                    TextView snippet = new TextView(getApplicationContext());
                    snippet.setTextColor(Color.GRAY);
                    snippet.setText(marker.getSnippet());

                    info.addView(title);
                    info.addView(snippet);

                    return info;
                }
            });
        } catch (Exception e) {
            Timber.e(e);
        }

    }

    /**
     * move camera to a position on map
     *
     * @param position  position is LatLng
     * @param zoomLevel zoom level on screen
     */
    public void showCameraToPosition(LatLng position, float zoomLevel) {
        try {
            CameraPosition cameraPosition = CameraPosition.builder()
                    .target(position)
                    .zoom(zoomLevel)
                    .bearing(0.0f)
                    .tilt(0.0f)
                    .build();

            if (mMap != null) {
                mMap.animateCamera(CameraUpdateFactory.newCameraPosition(cameraPosition), null);
            }
        } catch (Exception e) {
            Timber.e(e);
        }

    }

    /**
     * draw circle on positon of map
     *
     * @param position position is LatLng
     * @param radius   radius of circle
     */
    public void showCircleToGoogleMap(LatLng position, float radius, int color, String tag) {
        try {
            if (position == null) {
                return;
            }

            CircleOptions circleOptions = new CircleOptions();
            circleOptions.center(position);
            circleOptions.radius(radius * 1000);
            circleOptions.fillColor(getResources().getColor(color));
            circleOptions.strokeWidth(0);
            circleOptions.clickable(true);

            if (mMap != null) {
                Circle circle = mMap.addCircle(circleOptions);
                circle.setTag(tag);
            }
        } catch (Exception e) {
            Timber.e(e);
        }
    }

    /**
     * write time guess to firebase
     */
    private void writeTimeGuessToFirebase(String guessTime) {
        mRef.child("guess").child("is-guess").setValue(true);
        mRef.child("guess").child("guess-time").setValue(guessTime);
    }


}
